# Modified from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
FastInst criterion.
"""
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import(
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


from matplotlib import pyplot as plt

from detectron2.utils.comm import get_world_size
from torch import nn

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: number of masks
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def kd_loss(logits_student, logits_teacher, temperature=4):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


# kd_loss_jit = torch.jit.script(
#     kd_loss
# )  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)  # (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), tensor([ 4, 15, 74, 80, 24, 25, 26, 44, 47, 90, 92]))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # tensor([74, 56, 41, 56,  3, 39,  1,  7,  7,  5, 39], device='cuda:1')
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o  # 最终每个query的真值标签 (2, 100)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)  # (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), tensor([15, 37, 88, 95,  2, 20, 24, 26, 47, 67, 90]))
        tgt_idx = self._get_tgt_permutation_idx(indices)  # (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), tensor([0, 3, 2, 1, 1, 5, 0, 3, 2, 6, 4]))
        src_masks = outputs['pred_masks']  # (bs, 100, 92, 92)
        src_masks = src_masks[src_idx]  # (11, 92, 92)
        masks = [t["masks"] for t in targets]  # list:bs {(4, 736, 736)  (7, 736, 736)}
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()  # (2, 7, 736, 736)  (2, 736, 736)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]  # (11, 736, 736)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]  # (11, 1, 92, 92)
        target_masks = target_masks[:, None]  # (11, 1, 736, 736)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )  # (11, 12544, 2)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)  # (11, 12544)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # (11, 12544)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_distillation(self, outputs, indices):
        """Calculate distillation loss
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)  # (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), tensor([15, 37, 88, 95,  2, 20, 24, 26, 47, 67, 90]))
        src_masks = outputs['pred_masks']  # (bs, 100, 92, 92)
        src_masks = src_masks[src_idx]  # (11, 92, 92)

        return src_masks

    def loss_proposals(self, output_proposals, targets, indices):
        assert "proposal_cls_logits" in output_proposals

        proposal_size = output_proposals["proposal_cls_logits"].shape[-2:]  # [46, 46]
        proposal_cls_logits = output_proposals["proposal_cls_logits"].flatten(2).float()  # b, c, hw  (2, 81, 46*46)

        target_classes = self.num_classes * torch.ones([proposal_cls_logits.shape[0],
                                                        proposal_size[0] * proposal_size[1]],
                                                       device=proposal_cls_logits.device)  # (bs, h*w)
        target_classes = target_classes.to(torch.int64)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # tensor([56, 56, 41, 74,  7,  7,  5, 39,  3,  1, 39], device='cuda:1')
        idx = self._get_src_permutation_idx(indices)  # (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), tensor([ 468, 1169, 1179, 1707,  603,  700,  709,  962, 1019, 1082, 1132]))
        target_classes[idx] = target_classes_o

        loss_proposal = F.cross_entropy(proposal_cls_logits, target_classes, ignore_index=-1)
        losses = {"loss_proposal": loss_proposal}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def total_loss_dis(self, student_logits, teacher_logits, src_masks, gt_masks, targets, num_masks):
        src_masks = src_masks.unsqueeze(0)
        mask_h, mask_w = targets[0]['masks'].shape[1], targets[0]['masks'].shape[2]
        src_masks = F.interpolate(src_masks, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
        src_masks = src_masks.squeeze(0)

        gt_masks = (gt_masks < 0.).to(src_masks)
        gt_masks = gt_masks.unsqueeze(0)
        gt_masks = F.interpolate(gt_masks, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
        gt_masks = gt_masks.squeeze(0)

        src_masks = src_masks[:, None]
        gt_masks = gt_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )  # (11, 12544, 2)
            # get gt labels
            point_labels = point_sample(
                gt_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)  # (11, 12544)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # (11, 12544)

        l_dict = {
            "loss_kd": kd_loss(student_logits, teacher_logits.detach()),
            "loss_mask_dis": sigmoid_ce_loss_jit(point_logits, point_labels.detach(), num_masks),
            "loss_dice_dis": dice_loss_jit(point_logits, point_labels.detach(), num_masks),
        }

        return l_dict

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute proposal loss
        global finally_prediction_mask
        proposal_loss_dict = {}
        if outputs.get("proposal_cls_logits") is not None:
            output_proposals = {"proposal_cls_logits": outputs.pop("proposal_cls_logits")}
            indices = self.matcher(output_proposals, targets)
            proposal_loss_dict = self.loss_proposals(output_proposals, targets, indices)

        # Compute the main output loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if outputs_without_aux.get("pred_matching_indices") is not None:
            indices = outputs_without_aux["pred_matching_indices"]
        else:
            indices = self.matcher(outputs_without_aux, targets)  # [(tensor([ 4, 15, 74, 80]), tensor([1, 3, 2, 0])), (tensor([24, 25, 26, 44, 47, 90, 92]), tensor([0, 6, 3, 1, 2, 4, 5]))]

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)  #
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()  # 11.0

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if aux_outputs.get("pred_matching_indices") is not None:
                    indices = aux_outputs["pred_matching_indices"]
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses.update(proposal_loss_dict)

        # Knowledge distillation loss
        if "aux_outputs" in outputs:
            finally_prediction_mask = outputs.copy()
            student_logits = outputs["aux_outputs"][0]["pred_logits"].float().transpose(1, 2)
            src_masks = torch.zeros_like(outputs["aux_outputs"][0]["pred_masks"]).to(outputs["aux_outputs"][0]["pred_masks"])
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if i == 5:
                    student_logits = aux_outputs['pred_logits'].float().transpose(1, 2)  # (bs, 81, 100)
                    indices = self.matcher(aux_outputs, targets)  # ([0, 1, 18, 26], [0, 2, 1, 3])
                    src_masks = self.loss_masks_distillation(outputs=aux_outputs, indices=indices)  # (11, 92, 92)

                elif i == 6:
                    teacher_logits = aux_outputs['pred_logits'].float().transpose(1, 2)  # (bs, 81, 100)
                    indices = self.matcher(aux_outputs, targets)  # ([0, 1, 2, 8], [0, 2, 1, 3])  ([0, 4, 5, 8, 18, 22, 24], [1, 5, 2, 3, 4, 6, 0])
                    gt_masks = self.loss_masks_distillation(outputs=aux_outputs,  indices=indices)  # (11, h, w) (11, 92, 92)
                    # visual_gt_mask_all = gt_masks  # [11, 416, 576]
                    # for mask_item in range(gt_masks.shape[0]):
                    #     visual_gt_mask = visual_gt_mask_all[mask_item, :, :]
                    #     visual_gt_mask = visual_gt_mask.view(gt_masks.shape[1], gt_masks.shape[2])  # [416, 576]
                    #     visual_gt_mask = visual_gt_mask.cpu()
                    #     visual_gt_mask = visual_gt_mask.numpy()
                    #     # 将布尔类型的掩码数据转换为整数类型的图像数据
                    #     mask_data = np.where(visual_gt_mask, 0, 255).astype(np.uint8)
                    #     # 显示图像
                    #     plt.imshow(mask_data, cmap='gray')
                    #     plt.axis('off')
                    #     plt.suptitle('predict')
                    #     plt.savefig(fname="predict_mask", bbox_inches='tight', pad_inches=0)
                    #     plt.show()
                    # shen end
                    l_dict = self.total_loss_dis(student_logits, teacher_logits, src_masks, gt_masks, targets, num_masks)
                    l_dict = {k + f"_{0}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                elif i == 7:
                    student_logits = aux_outputs['pred_logits'].float().transpose(1, 2)
                    teacher_logits = finally_prediction_mask['pred_logits'].float().transpose(1, 2)

                    indices = self.matcher(aux_outputs, targets)
                    src_masks = self.loss_masks_distillation(outputs=aux_outputs, indices=indices)
                    indices = self.matcher(finally_prediction_mask, targets)
                    gt_masks = self.loss_masks_distillation(outputs=finally_prediction_mask, indices=indices)

                    l_dict = self.total_loss_dis(student_logits, teacher_logits, src_masks, gt_masks, targets, num_masks)
                    l_dict = {k + f"_{1}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                elif i == 8:
                    student_logits = aux_outputs['pred_logits'].float().transpose(1, 2)
                    teacher_logits = finally_prediction_mask['pred_logits'].float().transpose(1, 2)

                    indices = self.matcher(aux_outputs, targets)
                    src_masks = self.loss_masks_distillation(outputs=aux_outputs, indices=indices)
                    indices = self.matcher(finally_prediction_mask, targets)
                    gt_masks = self.loss_masks_distillation(outputs=finally_prediction_mask, indices=indices)

                    l_dict = self.total_loss_dis(student_logits, teacher_logits, src_masks, gt_masks, targets, num_masks)
                    l_dict = {k + f"_{2}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                    del finally_prediction_mask
                    del student_logits
                    del src_masks

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
