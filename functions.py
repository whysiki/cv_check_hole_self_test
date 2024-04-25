from PIL import Image, ImageOps
import os
import numpy as np
import cv2
from typing import Tuple, Union


class Morphology_Process:
    @staticmethod
    def close(
        image: Image.Image, structure_element_size: Tuple[int, int] = (5, 5)
    ) -> Image.Image:
        image_array = np.array(image)
        structure_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, structure_element_size
        )
        closed_image_array = cv2.morphologyEx(
            image_array, cv2.MORPH_CLOSE, structure_element
        )
        closed_image_pillow = Image.fromarray(closed_image_array)
        return closed_image_pillow

    @staticmethod
    def open(
        image: Image.Image, structure_element_size: Tuple[int, int] = (5, 5)
    ) -> Image.Image:
        image_array = np.array(image)
        structure_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, structure_element_size
        )
        closed_image_array = cv2.morphologyEx(
            image_array, cv2.MORPH_OPEN, structure_element
        )
        closed_image_pillow = Image.fromarray(closed_image_array)
        return closed_image_pillow


def crop_image_white_background(
    input_image: Union[str, Image.Image], show_image: bool = False
) -> Image.Image:
    if isinstance(input_image, Image.Image):
        image = input_image
    elif os.path.isfile(input_image):
        image = Image.open(input_image)
    else:
        raise ValueError("Invalid input image.")
    gray_image = image.convert("L")
    # gray_image = Morphology_Process.close(gray_image)
    threshold = 128
    binary_image = gray_image.point(lambda p: p > threshold and 255)
    binary_image_reversed = ImageOps.invert(binary_image)
    bbox: Tuple[int, int, int, int] = (
        binary_image_reversed.getbbox()
    )  # left, upper, right, lower
    # 稍微扩大一点
    extend_size = 10
    bbox = (
        bbox[0] - extend_size,
        bbox[1] - extend_size,
        bbox[2] + extend_size,
        bbox[3] + extend_size,
    )
    cropped_image = binary_image.crop(bbox)
    if show_image:
        cropped_image.show()
    return cropped_image


def k_re_class(input_image: str | Image.Image, k: int = 2):

    if isinstance(input_image, Image.Image):
        image = np.array(input_image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif os.path.isfile(input_image):
        # 读取图像并转换为灰度图
        image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Invalid input image.")

    # 将图像转换为一维数组
    data = image.flatten().reshape(-1, 1)

    data = np.float32(data)

    # 使用K均值聚类算法进行聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 将像素值重新分配到相应的类别中
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    # 显示原始图像和分割后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def auto_reclassify_and_binarize(
    image_path: str | Image.Image, num_classes=2
) -> Image.Image:

    # 读取图像并转换为灰度图像
    if isinstance(image_path, Image.Image):
        image = image_path
    elif os.path.isfile(image_path):
        image = Image.open(image_path)
    else:
        raise ValueError("Invalid input image.")

    gray_image = image.convert("L")

    # 将灰度图像转换为OpenCV格式
    gray_array = cv2.cvtColor(np.array(gray_image), cv2.COLOR_GRAY2BGR)

    # 将图像转换为一维数组
    data = gray_array.reshape((-1, 3)).astype(np.float32)

    # 定义要分成的类别数量
    k = num_classes

    # 使用K均值聚类算法进行聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 根据聚类结果对图像进行二值化处理
    segmented_image = (
        labels.reshape(gray_array.shape[0], gray_array.shape[1]).astype(np.uint8) * 255
    )

    resilt_image = Image.fromarray(segmented_image)

    # 判断是否需要反转颜色根据背景颜色, 如果背景颜色为黑色则反转颜色
    if np.mean(segmented_image) < 128:
        resilt_image = ImageOps.invert(resilt_image)

    return resilt_image
