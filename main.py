from functions import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import cpu_count
import concurrent.futures
from math import ceil, sqrt
import time


# 计算椭圆曲率
# def calculate_curvature(a, b, theta):

#     curvature = (
#         a * b / ((a**2 * np.sin(theta) ** 2 + b**2 * np.cos(theta) ** 2) ** (3 / 2))
#     )
#     return curvature


# 计算椭圆离心率
def calculate_eccentricity(a, b):
    a, b = max(a, b), min(a, b)
    round_score = 6
    a = round(a, round_score)
    b = round(b, round_score)
    a_2 = round(a**2, round_score)
    b_2 = round(b**2, round_score)

    e_ = sqrt(1 - b_2 / a_2)
    e_ = round(e_, round_score)
    # print("a:", a, "b:", b, "a^2:", a_2, "b^2:", b_2, "e:", e_)
    return e_


def single_image_processing(image_path: str, show_image=False):
    # 使用imageio库读取图像,将图像转换为灰度图像
    image: np.ndarray = imageio.imread(Path(image_path), mode="L")

    # 背景裁剪 + 二值化
    crop_image: np.ndarray = crop_image_white_background_opencv(
        image, show_image=show_image
    )

    print("crop_image.shape:", crop_image.shape)

    # 进行闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    crop_image = cv2.morphologyEx(crop_image, cv2.MORPH_CLOSE, kernel)

    # opencv的Canny边缘检测
    edges_image = cv2.Canny(crop_image, 100, 200)  # 100, 200

    # 再次进行闭运算
    # edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, kernel)

    print("edges_image.shape:", edges_image.shape)

    contours, _ = cv2.findContours(
        edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print("连通区域数量:", len(contours))

    if show_image:

        plt.figure()

        # 获取图像的宽度和高度 , opencv的shape是(h, w)（y, x）
        height, width = edges_image.shape

        # 设置图表刻度x轴和y轴的范围
        plt.xlim(0, width)
        plt.ylim(0, height)

    # 轮廓与拟合椭圆的相似度列表
    similarity_score_list = []

    # 椭圆离心率列表
    curvature_list = []

    # 计算各个连通区域的面积
    for contour in contours:

        # 拟合椭圆
        if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(contour)

            print("拟合椭圆:", ellipse)

            # # 计算离心率
            # 获取椭圆参数
            center, axes, angle = ellipse
            a, b = axes

            # 计算椭圆离心率
            eccentricity = calculate_eccentricity(a, b)

            curvature_list.append(eccentricity)

            print("椭圆离心率:", eccentricity)

            temp_image = np.array(edges_image)
            # 临时转换为cv2的BGR图像
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
            # 绘制椭圆
            cv2.ellipse(temp_image, ellipse, (0, 255, 0), 2)

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

            print("轮廓与拟合椭圆的相似度:", similarity_score)

            similarity_score_list.append(similarity_score)

            if show_image:

                cv2_imshow(temp_image, "Contours")

        # contour  轮廓点集 [x, y] x: 横坐标  y: 纵坐标 从左上角开始 -->x 横坐标递增 -->y 纵坐标递增
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(
            contour
        )  # x: 左上角横坐标 y: 左上角纵坐标 w: 宽度 h: 高度

        print("边界矩形:", x, y, w, h)

        if show_image:
            # 将NumPy数组转换为Python列表
            list_of_tuples = [tuple(subarr.flatten()) for subarr in contour]

            # print("轮廓点集:", list_of_tuples[:5], contour[:5])

            # 分解为两个列表：x坐标和y坐标
            x1, y1 = zip(*list_of_tuples)
            # 使用matplotlib绘制轮廓
            plt.plot(x1, y1)

            # #plt画出这个矩形
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])

    if show_image:
        # 反转y轴
        plt.gca().invert_yaxis()

        # 将x轴放到上方
        plt.gca().xaxis.tick_top()

        # 设置block=False以便程序继续执行

        plt.show(block=False)

        # 等待1秒后自动关闭
        plt.pause(1)
        plt.close()

    # 数据字典，连通区数量，各个连通区域的面积，边界矩形的坐标
    boundingRects = [cv2.boundingRect(contour) for contour in contours]

    def process_data(data, key, round_score=6):
        if len(data[key]) > 0:
            data[key] = [round(x, round_score) for x in data[key]]
            data[key + "_平均值"] = np.mean(data[key])
            data[key + "_最大值"] = np.max(data[key])
            data[key + "_最小值"] = np.min(data[key])
        else:
            data[key] = []
            data[key + "_平均值"] = 0
            data[key + "_最大值"] = 0
            data[key + "_最小值"] = 0

    data = {
        "连通区数量": len(contours),
        "各个连通区域的面积": [w * h for x, y, w, h in boundingRects],
        "各个边界矩形坐标": [(x, y, w, h) for x, y, w, h in boundingRects],
        "各个轮廓与拟合椭圆的相似度": similarity_score_list,
        "各个拟合椭圆离心率": curvature_list,
    }

    process_data(data, "各个连通区域的面积")
    process_data(data, "各个轮廓与拟合椭圆的相似度")
    process_data(data, "各个拟合椭圆离心率")

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
            executor.submit(single_image_processing, image_path, True)
            for image_path in image_path_list
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(image_path_list)
        ):
            datas.append(future.result())

    def process_dataframe(df: pd.DataFrame, filename: str) -> None:
        # 为df开头插入 从1开始，列名为序号 的列
        df.insert(0, "序号", range(1, len(df) + 1))

        # 尾部增加三行计算每列的平均， 最大， 最小
        # 计算每列的平均值、最大值和最小值，排序元素类型是列表的列,仅仅计算数值列
        # 选择数值列
        numeric_cols = df.select_dtypes(include="number")

        # 计算每列的平均值、最大值和最小值
        mean_values = numeric_cols.mean()
        max_values = numeric_cols.max()
        min_values = numeric_cols.min()

        # 将结果添加到 DataFrame 的尾部

        # 将结果添加到 DataFrame 的尾部
        df = pd.concat(
            [
                df,
                pd.DataFrame(mean_values).T,
                pd.DataFrame(max_values).T,
                pd.DataFrame(min_values).T,
            ],
            ignore_index=True,
        )

        # 数值列统一保留6位小数, 不足6位小数的补0
        nc = numeric_cols.columns.tolist()
        nc.remove("序号")
        for col in nc:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else x)

        # 重命名最后三行
        df.iloc[-3:, 0] = ["平均值", "最大值", "最小值"]

        df.to_excel(f"{filename}.xlsx", index=False)

    # 使用函数处理DataFrame
    process_dataframe(df=pd.DataFrame(datas), filename=os.path.basename(images))


if __name__ == "__main__":
    process_images(r"样本图像（正常及缺陷）\正常样本1")
    process_images(r"样本图像（正常及缺陷）\缺陷样本")
