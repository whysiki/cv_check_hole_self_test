from main import single_image_processing
from functions import *
from PIL import Image
from multiprocessing import Process

test_image_path1 = r"样本图像（正常及缺陷）\正常样本1\2.jpg"
test_image_path2 = r"2.jpg"


def crop_test():
    crop_image_white_background_opencv(test_image_path1, show_image=True)
    crop_image_white_background_opencv(test_image_path2, show_image=True)


def test_single_image_processing():
    single_image_processing(test_image_path1, show_image=True)
    single_image_processing(test_image_path2, show_image=True)


# print(image.size)

# crop_test()

test_single_image_processing()
