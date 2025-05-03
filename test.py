import numpy as np

def build_full_transition_matrix():
    """
    构造完整的 19x19 转移矩阵 P，状态编号约定：
      0     (吸收态: 等级跌出1以下 -> 0),
      1..17 (可活动的游戏状态, 等级1~17),
      18    (吸收态: 等级升到18).

    游戏规则：
      - 如果当前等级=1，再往下跳到0，则游戏结束(吸收)；
      - 如果当前等级=17，再往上跳到18，则游戏结束(吸收)；
      - 若 2..17 中 i>9，则 p_up=0.4, p_down=0.6；
      - 若 2..17 中 i<9，则 p_up=0.6, p_down=0.4；
      - i=9 时，p_up=p_down=0.5。
    """

    # 创建 19x19 的零矩阵
    P = np.zeros((19, 19), dtype=float)

    # 令 0 和 18 自身为吸收态 (自环)
    P[0, 0] = 1.0    # 等级=0
    P[18, 18] = 1.0  # 等级=18

    # 对于 i=1..17，计算转移概率
    for i in range(1, 18):
        if i > 9:
            p_up = 0.4
        elif i < 9:
            p_up = 0.6
        else:  # i == 9
            p_up = 0.5

        p_down = 1.0 - p_up

        # 上升1级
        if i == 17:
            # 若当前在17级，往上走则到18（吸收）
            P[i, 18] = p_up
        else:
            # 其他情况，往上走就到 i+1
            P[i, i + 1] = p_up

        # 下降1级
        if i == 1:
            # 若当前在1级，往下走则到0（吸收）
            P[i, 0] = p_down
        else:
            # 其他情况，往下走就到 i-1
            P[i, i - 1] = p_down

    return P


def compute_expected_time(Q):
    """
    给定暂态子矩阵 Q（对应状态1..17之间的相互转移，不含吸收态0和18），
    通过 (I - Q)^(-1) 求出基本矩阵 N，然后得到各状态的期望吸收时间。

    返回：一个长度17的向量 T,
          其中 T[i-1] = E[i], i=1..17
          即：从等级 i 出发，到到达0或18(吸收)的期望天数。
    """
    I = np.eye(Q.shape[0], dtype=float)
    N = np.linalg.inv(I - Q)             # 基本矩阵 N = (I - Q)^(-1)
    ones = np.ones((Q.shape[0], 1), dtype=float)
    t_vector = N @ ones                 # E = N * 1
    return t_vector.flatten()


def main():
    # 1) 构造完整的 19x19 转移矩阵 P
    P = build_full_transition_matrix()

    # 2) 从 P 中分离出 Q 子矩阵(对暂态状态 1..17)
    #    在 P 中，行列索引含义：
    #       0(吸收态0), 1..17(可活动等级1~17), 18(吸收态18)
    #    所以 Q = P[1..17, 1..17]   (numpy 切片 [start:end] 不含 end)
    Q = P[1:18, 1:18]

    # 3) 计算各状态到吸收(0或18)的期望时间
    T = compute_expected_time(Q)
    # 例如 T[0] = E[1], T[1] = E[2], ..., T[16] = E[17]

    # 我们关心从等级12出发 => i=12
    # 在 T 向量中下标 = (12 - 1) = 11
    E_12 = T[12 - 1]

    # 4) 计算 P(T > 480)
    #    初始分布 p(0)，长度=19，一开始全0，只在索引=12(等级12)处为1
    p0 = np.zeros(19, dtype=float)
    p0[12] = 1.0

    # P^480
    P_480 = np.linalg.matrix_power(P, 480)
    p_480 = p0 @ P_480

    # “尚未吸收”的概率就是处于索引=1..17 的总和
    prob_not_absorbed_480 = p_480[1:18].sum()

    # 5) 打印结果
    print("========== 结果输出 ==========")
    print(f"从等级12开始的期望吸收时间(天): {E_12:.6f}")
    print(f"游戏持续时间 > 480 天的概率: {prob_not_absorbed_480:.8f}")


if __name__ == "__main__":
    main()
