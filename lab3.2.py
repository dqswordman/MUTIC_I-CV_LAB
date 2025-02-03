# lab3.2
"""
数字减影血管造影(DSA)示例代码
在此代码中，我们读取两张图像(a)和(b)，分别称之为mask与live，
执行差值运算得到(c)，再对差值结果进行简单的直方图均衡以增强血管对比度，
模拟(d)增强后的差值图像。
"""

import cv2
import numpy as np


def main():
    # 1. 读取掩膜图像与实时图像（转为灰度）
    mask_img = cv2.imread("pic/pic_a.png", cv2.IMREAD_GRAYSCALE)
    live_img = cv2.imread("pic/pic_b.png", cv2.IMREAD_GRAYSCALE)

    # 2. 判断是否成功读取
    if mask_img is None or live_img is None:
        print("Error: 无法读取掩膜图像或实时图像，请检查文件路径！")
        return

    # 3. 计算差值图像 (c)
    # 使用绝对差值可避免简单相减带来的负值问题
    diff_img = cv2.absdiff(live_img, mask_img)

    # 4. 对差值图像进行简单的增强操作 (d)
    # 方法1：直方图均衡(灰度图常用)
    enhanced_diff = cv2.equalizeHist(diff_img)

    # 5. 显示并保存结果
    cv2.imshow("Mask Image (a)", mask_img)
    cv2.imshow("Live Image (b)", live_img)
    cv2.imshow("Difference Image (c)", diff_img)
    cv2.imshow("Enhanced Difference Image (d)", enhanced_diff)

    # 保存结果到硬盘（可选）
    cv2.imwrite("difference.jpg", diff_img)
    cv2.imwrite("enhanced_difference.jpg", enhanced_diff)

    # 按任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
