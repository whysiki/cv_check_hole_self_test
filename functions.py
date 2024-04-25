# from PIL import Image, ImageOps
# import os
# from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from typing import Tuple, Union
from pathlib import Path
import imageio
from pyautogui import size as screen_size


def crop_image_white_background_opencv(
    input_image: Union[str, np.ndarray], show_image: bool = False
) -> np.ndarray:

    if isinstance(input_image, np.ndarray):
        # 读取图像
        image = input_image
    elif isinstance(input_image, str):
        # 使用imageio库读取图像
        # image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        # 使用imageio库读取图像
        image = imageio.imread(Path(input_image), mode="L")

    else:

        raise ValueError("Invalid input image.")

    threshold = np.mean(image)

    ##二值化图像
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    ##反转图像,如果背景颜色为白色则反转颜色. 黑色为0, 白色为255
    if np.mean(binary_image) > 128:
        binary_image = cv2.bitwise_not(binary_image)

    # print("binary_image:", binary_image.tolist()[:])
    # 检查是否只有两种颜色
    # print("np.unique(binary_image):", np.unique(binary_image))

    # 找到非零像素的坐标
    # points = cv2.findNonZero(binary_image)

    # print("binary_image.shape:", binary_image.shape) # (h, w)

    # 计算边界框
    x, y, w, h = cv2.boundingRect(binary_image)

    # print("x, y, w, h:", x, y, w, h)

    # 稍微扩大一点
    extend_size = 10
    x = max(0, x - extend_size)
    y = max(0, y - extend_size)
    w = min(image.shape[1] - x, w + 2 * extend_size)
    h = min(image.shape[0] - y, h + 2 * extend_size)

    # 裁剪图像
    cropped_image = binary_image[y : y + h, x : x + w]

    # 显示图像
    if show_image:
        window_name = "Cropped Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # 获取屏幕尺寸， 把窗口大小设置为屏幕大小的50%，并移动到屏幕中间
        screen_width, screen_height = screen_size()

        cv2.resizeWindow(window_name, screen_width // 2, screen_height // 2)
        cv2.moveWindow(window_name, screen_width // 4, screen_height // 4)
        cv2.imshow(window_name, cropped_image)
        cv2.waitKey(1000 * 1)
        cv2.destroyAllWindows()

    return cropped_image


# 设置文件系统编码为UTF-8
# os.environ["PYTHONIOENCODING"] = "UTF-8"


# class Morphology_Process:
#     @staticmethod
#     def close(
#         image_array: np.ndarray, structure_element_size: Tuple[int, int] = (5, 5)
#     ) -> np.ndarray:
#         structure_element = cv2.getStructuringElement(
#             cv2.MORPH_RECT, structure_element_size
#         )
#         closed_image_array = cv2.morphologyEx(
#             image_array, cv2.MORPH_CLOSE, structure_element
#         )
#         return closed_image_array


# def auto_reclassify_and_binarize_opencv(
#     input_image: Union[str, np.ndarray], num_classes=2
# ) -> np.ndarray:

#     if isinstance(input_image, np.ndarray):
#         # 读取图像
#         image = input_image
#     elif isinstance(input_image, str):
#         # 使用imageio库读取图像
#         # image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
#         # 使用imageio库读取图像, mode="L"表示灰度图像
#         image = imageio.imread(Path(input_image), mode="L")

#     else:

#         raise ValueError("Invalid input image.")

#     # 将图像转换为一维数组
#     data = image.reshape((-1, 1)).astype(np.float32)

#     # 使用MiniBatchKMeans进行聚类
#     kmeans = MiniBatchKMeans(n_clusters=num_classes, batch_size=1000, max_iter=100)
#     labels = kmeans.fit_predict(image.reshape((-1, 1)))

#     # 根据聚类结果对图像进行二值化处理
#     segmented_image = (labels.reshape(image.shape) * 255 / (num_classes - 1)).astype(
#         np.uint8
#     )

#     # 判断是否需要反转颜色根据背景颜色, 如果背景颜色为黑色则反转颜色
#     if np.mean(segmented_image) < 128:
#         segmented_image = cv2.bitwise_not(segmented_image)

#     return segmented_image
