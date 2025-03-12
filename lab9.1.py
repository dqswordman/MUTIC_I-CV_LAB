import cv2
import numpy as np


def pseudocolor_world_map(input_path, output_path):
    """
    将世界地图灰度图进行伪彩色处理并保存结果
    :param input_path: 输入灰度图的文件路径，如 'pic/a.jpg'
    :param output_path: 输出伪彩色图的保存路径，如 'pic/a_pseudocolor.jpg'
    """
    # 1. 读取灰度图 (cv2.IMREAD_GRAYSCALE 指定以灰度模式读取)
    gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print("读取灰度图失败，请检查输入路径是否正确。")
        return

    # 2. 应用OpenCV自带的伪彩色映射
    # 常见的映射模式包括:
    #   cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_BONE, cv2.COLORMAP_RAINBOW 等
    # 这里选择 JET 映射举例
    color_map = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

    # 3. 保存生成的伪彩色图
    cv2.imwrite(output_path, color_map)

    # 4. 如果需要查看，可以用 OpenCV 的 imshow 或者其他显示库
    #    在脚本中直接显示：
    # cv2.imshow("Pseudocolor World Map", color_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 如果只是单独运行脚本，可以这样调用
if __name__ == "__main__":
    input_image_path = "pic/a.png"  # 世界地图灰度图的路径
    output_image_path = "pic/a_pseudocolor.png"  # 输出的伪彩色图
    pseudocolor_world_map(input_image_path, output_image_path)
    print("伪彩色处理完成！已保存到：", output_image_path)
