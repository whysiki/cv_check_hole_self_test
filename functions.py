# from PIL import Image, ImageOps
# import os
# from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
from typing import Tuple, Union
from pathlib import Path
import imageio
from pyautogui import size as screen_size


def cv2_imshow(image: np.ndarray, window_name: str = "Image") -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 获取屏幕尺寸， 把窗口大小设置为屏幕大小的50%，并移动到屏幕中间
    screen_width, screen_height = screen_size()

    cv2.resizeWindow(window_name, screen_width // 2, screen_height // 2)
    cv2.moveWindow(window_name, screen_width // 4, screen_height // 4)
    cv2.imshow(window_name, image)
    cv2.waitKey(1000 * 1)
    cv2.destroyAllWindows()


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

    # 稍微缩小一点
    extend_size = -(binary_image.shape[0] // 7)
    x = max(0, x - extend_size)
    y = max(0, y - extend_size)
    w = min(image.shape[1] - x, w + 2 * extend_size)
    h = min(image.shape[0] - y, h + 2 * extend_size)

    # 裁剪图像
    cropped_image = binary_image[y : y + h, x : x + w]

    # 显示图像
    if show_image:
        window_name = "Cropped Image"
        cv2_imshow(cropped_image, window_name)

    return cropped_image
