def center_crop(image, desired_width, desired_height):
    """Center_crop."""
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    crop_x1 = center_x - desired_width // 2
    crop_y1 = center_y - desired_height // 2
    crop_x2 = center_x + desired_width // 2
    crop_y2 = center_y + desired_height // 2
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return cropped_image