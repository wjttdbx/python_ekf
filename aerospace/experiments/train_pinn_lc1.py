"""
3D PINN 重训练实验：lambda_ctrl = 1.0（对照原始 0.05）
输出目录: checkpoints/sdre_pinn_lc1
"""

from aerospace.pinn.pinn_trainer import TrainConfig, train_pinn

config = TrainConfig(
    output_dir="checkpoints/sdre_pinn_lc1",
    lambda_ctrl=1.0,   # 原始值 0.05，此处放大 20x 让 ctrl loss 与 phys 贡献相当
    lambda_phys=0.2,   # 保持不变
)

if __name__ == "__main__":
    result = train_pinn(config)
    print("Done:", result)
