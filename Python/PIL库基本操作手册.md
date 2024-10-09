PIL（Python Imaging Library）是一个强大的图像处理库，但目前它的开发已经停止，继承它的是Pillow库。Pillow是PIL的一个分支，提供了与PIL相同的功能并进行了扩展和维护。以下是Pillow库的安装和一些常用方法的介绍。

安装Pillow
可以使用pip来安装Pillow：
```bash
pip install Pillow
```

### 常用方法介绍
1. 导入库
```python
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

image = Image.open("example.jpg")
print(image.format, image.size, image.mode)

box = (100, 100, 400, 400)  # (left, upper, right, lower)
cropped_image = image.crop(box)

rotated_image = image.rotate(45) # 负数为顺时针，正数为逆时针

blurred_image = image.filter(ImageFilter.BLUR)

resized_image = image.resize((100, 191))

enhancer_Contrast = ImageEnhance.Contrast(image)
enhanced_image = enhancer.enhance(0.8)

enhancer_Contrast = ImageEnhance.Contrast(image)
enhanced_image = enhancer.enhance(0.8)

enhancer_Color = ImageEnhance.Color(image)
enhanced_image = enhancer.enhance(2.0)

image.save("output.png")
```
原图 -> crop -> rotate -> filter -> resize -> enhancer_Contrast -> enhancer_Color

![PIL](figures/PIL1.png)

2. 显示图像
```python
image.show()
```
- show：在默认的图像查看器中显示图像。
3. 图像转换
# 转换为灰度图像
gray_image = image.convert("L")
- convert：转换图像模式。
4.  图像滤镜
blurred_image = image.filter(ImageFilter.BLUR)
- filter：对图像应用滤镜，常用滤镜包括ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.DETAIL等。
5.  图像增强
enhancer = ImageEnhance.Contrast(image)
enhanced_image = enhancer.enhance(2.0)
- ImageEnhance：用于增强图像的模块，可以增强对比度、亮度、颜色和锐度。
6.  图像合并
```python
new_image = Image.new("RGB", image.size)

# 粘贴两个图像到新的图像中
new_image.paste(image, (0, 0))
new_image.paste(rotated_image, (500, 908))
new_image.save("new_image.png")
```
![](figures/PIL2.png)
- new：创建一个新的图像。
- paste：将一个图像粘贴到另一个图像上。
这些是Pillow库的一些常用方法，通过这些方法，可以方便地对图像进行处理、转换和增强。如果需要更多高级功能，可以参考Pillow的官方文档：https://pillow.readthedocs.io/
