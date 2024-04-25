from main import single_image_processing
from functions import (
    k_re_class,
    auto_reclassify_and_binarize,
    crop_image_white_background,
    Morphology_Process,
)
from PIL import Image
from multiprocessing import Process

test_image_path1 = r"样本图像（正常及缺陷）\正常样本1\2.jpg"
test_image_path2 = r"2.jpg"
# data = single_image_processing(test_image_path, show_image=True, threshold=128)

# print(data)


# 读取图像

# image = Image.open(test_image_path)
# print(image.size)
# image.show()
# # # 灰度化
# gray_image = image.convert("L")
# 进行k-means分类
image = auto_reclassify_and_binarize(test_image_path1)
image = crop_image_white_background(image)
image.show()


image = auto_reclassify_and_binarize(test_image_path2)
image = crop_image_white_background(image)
image.show()
# print(image.size)
