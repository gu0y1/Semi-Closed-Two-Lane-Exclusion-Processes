import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# ===== 全局参数 =====
L = 500
N = 250
move_A_rate = 0.8
A_to_B_rate = 0.01
move_B_rate = 0.6
B_to_A_rate = 0.01
steps = 50000
burning_steps = 10000

# ===== prll =====
def simulate_fully_parallel(entry_rate, exit_B_rate):
    A = np.zeros(L, dtype=int)
    B = np.zeros(L, dtype=int)

    # 初始化：随机分配粒子
    positions = np.random.choice(L * 2, N, replace=False)
    for pos in positions:
        if pos < L:
            A[pos] = 1
        else:
            B[pos - L] = 1

    A_count = np.zeros(L, dtype=int)
    B_count = np.zeros(L, dtype=int)

    for t in trange(steps, desc="Simulating", leave=False):
        A_next = A.copy()
        B_next = B.copy()
        actions = {}

        # 内部移动 + 交换提案
        for i in range(L):
            if i < L - 1 and A[i] == 1 and A[i + 1] == 0 and np.random.rand() < move_A_rate:
                actions[("A", i)] = ("A", i + 1)
            if i > 0 and B[i] == 1 and B[i - 1] == 0 and np.random.rand() < move_B_rate:
                actions[("B", i)] = ("B", i - 1)
            if A[i] == 1 and B[i] == 0 and np.random.rand() < A_to_B_rate:
                actions[("A", i)] = ("B", i)
            if B[i] == 1 and A[i] == 0 and np.random.rand() < B_to_A_rate:
                actions[("B", i)] = ("A", i)

        # 冲突检测
        occupied_targets = set()
        accepted_actions = []

        for source, target in actions.items():
            if target not in occupied_targets:
                occupied_targets.add(target)
                accepted_actions.append((source, target))

        # 应用动作
        for (from_lane, i), (to_lane, j) in accepted_actions:
            if from_lane == "A":
                A_next[i] = 0
            else:
                B_next[i] = 0

            if to_lane == "A":
                A_next[j] = 1
            else:
                B_next[j] = 1

        # 入口和出口
        if np.random.rand() < entry_rate and A[0] == 0:
            A_next[0] = 1
        if B[0] == 1 and np.random.rand() < exit_B_rate:
            B_next[0] = 0

        # 更新状态
        A[:] = A_next
        B[:] = B_next

        # 密度记录
        if t >= burning_steps:
            A_count += A
            B_count += B

    avg_steps = steps - burning_steps
    return A_count / avg_steps, B_count / avg_steps

# ===== 配置参数 =====
configs = {
    "(a) β=0.05": {
        "vary": "α",
        "entry_rates": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "exit_B_rate": 0.05,
    },
    "(b) α=0.35": {
        "vary": "β",
        "entry_rates": [0.35],
        "exit_B_rates": [0.20, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.50],
    },
    "(c) α=0.7": {
        "vary": "β",
        "entry_rates": [0.7],
        "exit_B_rates": [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    },
    "(d) β=0.4": {
        "vary": "α",
        "entry_rates": [0, 0.15, 0.30, 0.45, 0.50, 0.53, 0.56, 0.60, 0.75, 0.90],
        "exit_B_rate": 0.4,
    },
}

# ===== 绘图结构 =====
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for group_idx, (title, config) in enumerate(configs.items()):
    ax_A = axes[group_idx * 2]
    ax_B = axes[group_idx * 2 + 1]

    if config["vary"] == "α":
        for alpha in config["entry_rates"]:
            print(f"Simulating {title} α = {alpha}")
            rho_A, rho_B = simulate_fully_parallel(entry_rate=alpha, exit_B_rate=config["exit_B_rate"])
            ax_A.plot(range(1, L + 1), rho_A, label=f"α = {alpha}")
            ax_B.plot(range(1, L + 1), rho_B, label=f"α = {alpha}")
    else:
        for beta in config["exit_B_rates"]:
            print(f"Simulating {title} β = {beta}")
            rho_A, rho_B = simulate_fully_parallel(entry_rate=config["entry_rates"][0], exit_B_rate=beta)
            ax_A.plot(range(1, L + 1), rho_A, label=f"β = {beta}")
            ax_B.plot(range(1, L + 1), rho_B, label=f"β = {beta}")

    ax_A.set_title(f"{title} - Lane A", fontsize=14)
    ax_A.set_ylabel("ρ", fontsize=12)
    ax_A.set_ylim(0, 1)
    ax_A.grid(False)
    ax_A.legend()

    ax_B.set_title(f"{title} - Lane B", fontsize=14)
    ax_B.set_ylim(0, 1)
    ax_B.grid(False)
    ax_B.legend()

for ax in axes[-2:]:
    ax.set_xlabel("site i", fontsize=12)

plt.tight_layout()
plt.show()
