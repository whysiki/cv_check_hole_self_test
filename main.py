from PIL import Image, ImageFilter, ImageDraw
from functions import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import cpu_count
import concurrent.futures
from math import ceil


def single_image_processing(image_path: str, show_image=False):
    # 使用imageio库读取图像,将图像转换为灰度图像
    image: np.ndarray = imageio.imread(Path(image_path), mode="L")

    # 背景裁剪 + 二值化
    crop_image: np.ndarray = crop_image_white_background_opencv(
        image, show_image=show_image
    )

    # opencv的Canny边缘检测
    edges_image = cv2.Canny(crop_image, 100, 200)

    contours, _ = cv2.findContours(
        edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # print("连通区域数量:", len(contours))

    if show_image:

        plt.figure()

        # 获取图像的宽度和高度 , opencv的shape是(h, w)
        height, width = edges_image.shape

        # 设置图表刻度x轴和y轴的范围
        plt.xlim(0, width)
        plt.ylim(0, height)

    # 轮廓与拟合椭圆的相似度列表
    similarity_score_list = []

    # 计算各个连通区域的面积
    for contour in contours:

        # 拟合椭圆
        if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            temp_image = np.array(edges_image)
            # 绘制椭圆
            cv2.ellipse(temp_image, ellipse, (0, 255, 0), 2)

            # 判断椭圆是否闭合且光滑
            # is_closed_smooth = cv2.isContourConvex(contour)

            # 从椭圆对象中提取其边界点
            points = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                int(ellipse[2]),
                0,
                360,
                10,
            )

            # 将点集转换为numpy数组
            points = np.array(points)

            # 计算椭圆的Hu矩
            ellipse_moments = cv2.moments(points)
            ellipse_hu_moments = cv2.HuMoments(ellipse_moments).flatten()

            # 计算轮廓的Hu矩
            contour_moments = cv2.moments(contour)
            contour_hu_moments = cv2.HuMoments(contour_moments).flatten()

            # 计算两者之间的相似度分数
            similarity_score = cv2.matchShapes(
                ellipse_hu_moments, contour_hu_moments, cv2.CONTOURS_MATCH_I1, 0
            )  # cv2.CONTOURS_MATCH_I1方法 0-1之间的值，0表示完全匹配，>1表示不匹配

            # print("轮廓与拟合椭圆的相似度:", similarity_score)

            similarity_score_list.append(similarity_score)

            if show_image:
                # 显示结果
                # 创建一个窗口
                cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)

                # 设置窗口的大小
                # cv2.resizeWindow("Contours", 600, 600)

                # 在窗口中显示图像
                cv2.imshow("Contours", temp_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # 将NumPy数组转换为Python列表
        list_of_tuples = [tuple(subarr.flatten()) for subarr in contour]
        # print("轮廓点集:", list_of_tuples[:5], contour[:5])

        # 分解为两个列表：x坐标和y坐标
        x1, y1 = zip(*list_of_tuples)

        # contour  轮廓点集 [x, y] x: 横坐标  y: 纵坐标 从左上角开始 -->x 横坐标递增 -->y 纵坐标递增
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(
            contour
        )  # x: 左上角横坐标 y: 左上角纵坐标 w: 宽度 h: 高度

        # print("边界矩形:", x, y, w, h)

        if show_image:
            # 使用matplotlib绘制轮廓
            plt.plot(x1, y1)

            # #plt画出这个矩形
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])

        # 计算边界框的面积
        # area = w * h
        # print("面积:", area)

    if show_image:
        # 反转y轴
        plt.gca().invert_yaxis()

        # 将x轴放到上方
        plt.gca().xaxis.tick_top()
        # 显示图像
        plt.show()

    # 数据字典，连通区数量，各个连通区域的面积，边界矩形的坐标
    boundingRects = [cv2.boundingRect(contour) for contour in contours]
    data = {
        "连通区数量": len(contours),
        "各个连通区域的面积": [w * h for x, y, w, h in boundingRects],
        # "各个边界矩形坐标": [(x, y, w, h) for x, y, w, h in boundingRects],
        "各个轮廓与拟合椭圆的相似度": similarity_score_list,
    }

    # 保留五位小数
    data["各个连通区域的面积"] = [round(x, 5) for x in data["各个连通区域的面积"]]
    data["各个轮廓与拟合椭圆的相似度"] = [
        round(x, 5) for x in data["各个轮廓与拟合椭圆的相似度"]
    ]

    return data


# 读取文件夹内的所有图片， 返回路径列表
def read_all_image(path):
    path_list = []
    for file in tqdm(os.listdir(path)):
        if file.endswith(".jpg") or file.endswith(".png"):
            path_list.append(os.path.join(path, file))

    return path_list


def process_images(images: str) -> None:
    image_path_list = read_all_image(images)
    datas = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=ceil(cpu_count() / 2)
    ) as executor:
        futures = {
            executor.submit(single_image_processing, image_path, False)
            for image_path in image_path_list
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(image_path_list)
        ):
            datas.append(future.result())

    # 保存为xlsx文件
    df = pd.DataFrame(datas)
    df.to_excel(f"{os.path.basename(images)}.xlsx", index=False)


if __name__ == "__main__":
    process_images(r"样本图像（正常及缺陷）\正常样本1")
    process_images(r"样本图像（正常及缺陷）\缺陷样本")


#
#
#

# def single_image_processing(
#     image_path: str, show_image=False, threshold=128, auto_threshold=False
# ):  # , threshold=128):
#     # 打开图像文件
#     # image = Image.open(image_path)

#     # 将图像转换为灰度图像
#     # gray_image = image.convert("L")

#     # 形态学闭合操作
#     # gray_image = Morphology_Process.close(gray_image)

#     # 对灰度图像进行二值化

#     # if not auto_threshold:
#     #     binary_image = gray_image.point(lambda p: p > threshold and 255)

#     #     # print(type(binary_image))  # <class 'PIL.Image.Image'>
#     # else:
#     #     # 使用k-means算法对图像进行自动分类和二值化, 并自动反转颜色
#     #     binary_image = auto_reclassify_and_binarize(gray_image)

#     # # 裁剪图像外部的白色背景
#     # crop_image = crop_image_white_background(binary_image, show_image=show_image)

#     # edges_image = crop_image.filter(ImageFilter.FIND_EDGES)

#     # 查找图像中的连通区域
#     # contours, _ = cv2.findContours(
#     #     np.array(edges_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     # )

#     # 使用imageio库读取图像,将图像转换为灰度图像
#     image: np.ndarray = imageio.imread(Path(image_path), mode="L")

#     # 背景裁剪 + 二值化
#     crop_image: np.ndarray = crop_image_white_background_opencv(
#         image, show_image=show_image
#     )

#     # opencv的Canny边缘检测
#     edges_image = cv2.Canny(crop_image, 100, 200)

#     contours, _ = cv2.findContours(
#         edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # print("连通区域数量:", len(contours))

#     if show_image:

#         # 转换图像为opencv彩图
#         edges_image = cv2.cvtColor(np.array(edges_image), cv2.COLOR_GRAY2BGR)

#         # 重新转换为PIL图像
#         edges_image = Image.fromarray(edges_image)

#         # 创建一个 ImageDraw 对象
#         draw = ImageDraw.Draw(edges_image)

#         plt.figure()

#         # 获取图像的宽度和高度
#         width, height = edges_image.size

#         # 设置图表刻度x轴和y轴的范围
#         plt.xlim(0, width)
#         plt.ylim(0, height)

#     # 轮廓与拟合椭圆的相似度列表
#     similarity_score_list = []

#     # 计算各个连通区域的面积
#     for contour in contours:

#         # 拟合椭圆
#         if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
#             ellipse = cv2.fitEllipse(contour)
#             temp_image = np.array(edges_image)
#             # 绘制椭圆
#             cv2.ellipse(temp_image, ellipse, (0, 255, 0), 2)

#             # 判断椭圆是否闭合且光滑
#             # is_closed_smooth = cv2.isContourConvex(contour)

#             # 从椭圆对象中提取其边界点
#             points = cv2.ellipse2Poly(
#                 (int(ellipse[0][0]), int(ellipse[0][1])),
#                 (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
#                 int(ellipse[2]),
#                 0,
#                 360,
#                 10,
#             )

#             # 将点集转换为numpy数组
#             points = np.array(points)

#             # 计算椭圆的Hu矩
#             ellipse_moments = cv2.moments(points)
#             ellipse_hu_moments = cv2.HuMoments(ellipse_moments).flatten()

#             # 计算轮廓的Hu矩
#             contour_moments = cv2.moments(contour)
#             contour_hu_moments = cv2.HuMoments(contour_moments).flatten()

#             # 计算两者之间的相似度分数
#             similarity_score = cv2.matchShapes(
#                 ellipse_hu_moments, contour_hu_moments, cv2.CONTOURS_MATCH_I1, 0
#             )  # cv2.CONTOURS_MATCH_I1方法 0-1之间的值，0表示完全匹配，>1表示不匹配

#             # print("轮廓与拟合椭圆的相似度:", similarity_score)

#             similarity_score_list.append(similarity_score)

#             if show_image:
#                 # 显示结果
#                 # 创建一个窗口
#                 cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)

#                 # 设置窗口的大小
#                 # cv2.resizeWindow("Contours", 600, 600)

#                 # 在窗口中显示图像
#                 cv2.imshow("Contours", temp_image)

#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()

#         # 将NumPy数组转换为Python列表
#         list_of_tuples = [tuple(subarr.flatten()) for subarr in contour]
#         # print("轮廓点集:", list_of_tuples[:5], contour[:5])

#         # 分解为两个列表：x坐标和y坐标
#         x1, y1 = zip(*list_of_tuples)

#         # contour  轮廓点集 [x, y] x: 横坐标  y: 纵坐标 从左上角开始 -->x 横坐标递增 -->y 纵坐标递增
#         # 获取边界矩形
#         x, y, w, h = cv2.boundingRect(
#             contour
#         )  # x: 左上角横坐标 y: 左上角纵坐标 w: 宽度 h: 高度

#         # print("边界矩形:", x, y, w, h)

#         if show_image:
#             # 使用matplotlib绘制轮廓
#             plt.plot(x1, y1)

#             # #plt画出这个矩形
#             plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])

#         if show_image:

#             # 标记边界框
#             draw.rectangle([x, y, x + w, y + h], outline="red")

#         # 计算边界框的面积
#         # area = w * h
#         # print("面积:", area)

#     if show_image:
#         # 反转y轴
#         plt.gca().invert_yaxis()

#         # 将x轴放到上方
#         plt.gca().xaxis.tick_top()
#         # 显示图像
#         # plt.show()
#         # 显示标记后的图像
#         edges_image.show()

#     # 数据字典，连通区数量，各个连通区域的面积，边界矩形的坐标
#     boundingRects = [cv2.boundingRect(contour) for contour in contours]
#     data = {
#         "连通区数量": len(contours),
#         "各个连通区域的面积": [w * h for x, y, w, h in boundingRects],
#         # "各个边界矩形坐标": [(x, y, w, h) for x, y, w, h in boundingRects],
#         "各个轮廓与拟合椭圆的相似度": similarity_score_list,
#     }

#     # 保留五位小数
#     data["各个连通区域的面积"] = [round(x, 5) for x in data["各个连通区域的面积"]]
#     data["各个轮廓与拟合椭圆的相似度"] = [
#         round(x, 5) for x in data["各个轮廓与拟合椭圆的相似度"]
#     ]

#     return data
