"""单样本调试：检查 PINN 和 CARE Eigendecomp 的问题"""

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from aerospace.pinn.data_generator import extract_asdc_features, _solve_are_balanced
from aerospace.pinn.checkpoint_utils import normalize_pinn_state_dict
from aerospace.pinn.pinn_trainer import SDREPINN
from aerospace.paths import CHECKPOINTS_DIR

# 物理参数
MU, A_C, E_C = 3.986e5, 15000.0, 0.5
GAMMA = np.sqrt(2.0)
STATE_DIM, CTRL_DIM = 6, 3

Q = np.eye(STATE_DIM) * 0.001
R = np.eye(CTRL_DIM) * 1e7
B_P = np.zeros((STATE_DIM, CTRL_DIM))
B_P[3, 0] = B_P[4, 1] = B_P[5, 2] = 1.0
B_E = -B_P

GAM2 = GAMMA ** 2
R_INV = np.linalg.inv(R)
S_MAT = (1.0 - 1.0 / GAM2) * (B_P @ R_INV @ B_P.T)

def care_residual(A, P):
    R_ = A.T @ P + P @ A - P @ S_MAT @ P + Q
    return float(np.linalg.norm(R_, "fro"))

# 生成一个样本
pc = A_C * (1 - E_C**2)
x_p = np.array([500, 500, 500, 0, 0, 0])
x_e = np.array([0, 0, 0, 0, 0, 0])
nu = 0.0
rc = pc / (1 + E_C * np.cos(nu))
h = np.sqrt(MU * pc)
nd = h / rc**2
ndd = -2*MU*E_C*np.sin(nu) / rc**3 * (h / (MU*(1+E_C*np.cos(nu))))

# 构造 A_SDC
A = np.zeros((6, 6))
A[0, 3] = A[1, 4] = A[2, 5] = 1.0
rp = np.sqrt((rc + x_p[0])**2 + x_p[1]**2 + x_p[2]**2)
re = np.sqrt((rc + x_e[0])**2 + x_e[1]**2 + x_e[2]**2)
dx, dy, dz = x_p[0]-x_e[0], x_p[1]-x_e[1], x_p[2]-x_e[2]
r2 = dx*dx + dy*dy + dz*dz + 1e-6
bx = -MU*(rc+x_p[0])/rp**3 + MU*(rc+x_e[0])/re**3
by = -MU*x_p[1]/rp**3 + MU*x_e[1]/re**3
bz = -MU*x_p[2]/rp**3 + MU*x_e[2]/re**3
A[3, 0] = nd**2 + bx*dx/r2;  A[3, 1] = ndd + bx*dy/r2;  A[3, 2] = bx*dz/r2
A[4, 0] = -ndd + by*dx/r2;   A[4, 1] = nd**2 + by*dy/r2; A[4, 2] = by*dz/r2
A[5, 0] = bz*dx/r2;          A[5, 1] = bz*dy/r2;         A[5, 2] = bz*dz/r2
A[3, 4] = 2*nd;  A[4, 3] = -2*nd

print("=" * 70)
print("  单样本调试：θ=0, x_p=(500,500,500,0,0,0), x_e=zeros")
print("=" * 70)

# scipy 基准
P_scipy = _solve_are_balanced(A, B_P, Q, R / (1.0 - 1.0/GAM2))
res_scipy = care_residual(A, P_scipy)
print(f"\n[scipy 基准]")
print(f"  CARE 残差: {res_scipy:.3e}")
print(f"  P 范数: {np.linalg.norm(P_scipy, 'fro'):.3e}")
print(f"  P[0,0]: {P_scipy[0,0]:.6e}")
print(f"  P[3,3]: {P_scipy[3,3]:.6e}")

# PINN
ckpt_path = str(CHECKPOINTS_DIR / "sdre_pinn" / "best_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = SDREPINN(
    in_dim=int(ckpt["in_dim"]),
    backbone_dim=int(ckpt.get("backbone_dim", 256)),
    n_resblocks=int(ckpt.get("n_resblocks", 3)),
    head_hidden=int(ckpt.get("head_hidden", 64)),
    activation=str(ckpt.get("activation", "mish")),
)
sd = normalize_pinn_state_dict(ckpt["model_state_dict"])
model.load_state_dict(sd, strict=True)
model.eval()

feat = extract_asdc_features(A)
feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float64)
feat_std = np.asarray(ckpt["feat_std"], dtype=np.float64)
feat_std = np.where(feat_std < 1e-12, 1.0, feat_std)
l_mean = np.asarray(ckpt["l_mean"], dtype=np.float64)
l_std = np.asarray(ckpt["l_std"], dtype=np.float64)
l_std = np.where(l_std < 1e-12, 1.0, l_std)
delta_spd = float(ckpt["delta_spd"])

print(f"\n[特征提取]")
print(f"  feat_raw: {feat}")
feat_norm = (feat - feat_mean) / feat_std
print(f"  feat_norm: {feat_norm}")
print(f"  feat_mean: {feat_mean}")
print(f"  feat_std: {feat_std}")

x = torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    l_raw = model(x).squeeze(0).numpy().astype(np.float64)
l_vec = l_raw * l_std + l_mean

print(f"\n[PINN 输出]")
print(f"  l_raw (前5): {l_raw[:5]}")
print(f"  l_vec (前5): {l_vec[:5]}")
print(f"  l_mean (前5): {l_mean[:5]}")
print(f"  l_std (前5): {l_std[:5]}")

# 重建 P
tril_r, tril_c = np.tril_indices(STATE_DIM)
diag_mask = tril_r == tril_c
L = np.zeros((STATE_DIM, STATE_DIM))
for k, (r, c) in enumerate(zip(tril_r, tril_c)):
    L[r, c] = np.exp(l_vec[k]) if diag_mask[k] else l_vec[k]
P_pinn = L @ L.T + np.eye(STATE_DIM) * delta_spd

res_pinn = care_residual(A, P_pinn)
print(f"\n[PINN 结果]")
print(f"  CARE 残差: {res_pinn:.3e}")
print(f"  P 范数: {np.linalg.norm(P_pinn, 'fro'):.3e}")
print(f"  P[0,0]: {P_pinn[0,0]:.6e}")
print(f"  P[3,3]: {P_pinn[3,3]:.6e}")
print(f"  delta_spd: {delta_spd:.3e}")

print(f"\n[对比]")
print(f"  残差比 (PINN/scipy): {res_pinn/res_scipy:.3e}")
print(f"  P 范数比: {np.linalg.norm(P_pinn, 'fro') / np.linalg.norm(P_scipy, 'fro'):.3e}")
print(f"  P 差异范数: {np.linalg.norm(P_pinn - P_scipy, 'fro'):.3e}")

print("\n" + "=" * 70)
