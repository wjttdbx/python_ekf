"""
E2c-A 分析：标签计算、指标滚动统计、阈值选择、验证评估

seeds 0-4 → 训练（阈值选择）
seeds 5-9 → 验证（TPR/FPR/lead time）
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2

warnings.filterwarnings("ignore")

from aerospace.paths import DATA_DIR

# ═══════════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════════

CHI2_NEES_99 = chi2.ppf(0.99, 6)       # 16.81
CHI2_NIS_99 = chi2.ppf(0.99, 2)         # 9.21
LABEL_CONSECUTIVE = 10                   # 失配确认连续步数
NIS_WINDOW = 20                          # NIS 滚动窗口
LIN_GAP_WINDOW = 20                      # 线性化差距滚动窗口
P_COND_WINDOW = 20                       # P 条件数滚动窗口

GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]
SIGMA_VALUES = [0.001, 0.008, 0.1]

INDICATOR_NAMES = [
    "nis_frac",           # 滚动窗口内 NIS > 9.21 的比例
    "lin_gap_mean",       # 归一化测量均值线性化差距
    "lin_gap_cov",        # 归一化测量协方差线性化差距
    "log10_P_cond",       # log10(P 条件数)
    "neg_log10_P_eig_min", # -log10(P 最小特征值)
    "lin_gap_az",         # 方位角方向归一化差距
    "lin_gap_el",         # 俯仰角方向归一化差距
]


# ═══════════════════════════════════════════════════════════════════════════════
# 标签计算（离线，可用真值）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryLabels:
    geom: str
    seed: int
    sigma_deg: float
    n_steps: int
    nees: np.ndarray          # (n_steps,)
    has_mismatch: bool
    onset_step: int | None    # 失配起点步数（0-indexed），None 表示无失配
    onset_time_s: float | None
    mismatch_duration: int    # 失配持续步数
    peak_nees: float


def compute_labels(t: np.ndarray, nees: np.ndarray) -> TrajectoryLabels:
    """从 NEES 时序计算失配标签。

    规则：滚动窗口 NEES 连续 >= LABEL_CONSECUTIVE 步超出 CHI2_NEES_99。
    onset_step 为该窗口的第一步（首次指标超出时）。
    """
    mask = nees > CHI2_NEES_99
    n = len(mask)

    # 寻找第一个符合条件的连续段
    onset_step = None
    mismatch_duration = 0
    run_start = -1

    for i in range(n):
        if mask[i] and not np.isnan(nees[i]):
            if run_start < 0:
                run_start = i
            run_len = i - run_start + 1
            if run_len >= LABEL_CONSECUTIVE and onset_step is None:
                onset_step = run_start
        else:
            if onset_step is not None and run_start >= 0:
                mismatch_duration = max(mismatch_duration, i - onset_step)
            run_start = -1

    # 处理末尾还在失配中的情况
    if onset_step is not None and run_start >= 0:
        mismatch_duration = max(mismatch_duration, n - onset_step)

    has_mismatch = onset_step is not None
    onset_time = float(t[onset_step]) if has_mismatch else None

    # 找到峰值 NEES（仅在失配期间有意义，否则全局）
    peak_nees = float(np.nanmax(nees))

    return TrajectoryLabels(
        geom="", seed=-1, sigma_deg=0.0,
        n_steps=n, nees=nees,
        has_mismatch=has_mismatch,
        onset_step=onset_step,
        onset_time_s=onset_time,
        mismatch_duration=mismatch_duration,
        peak_nees=peak_nees,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 滚动指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_indicators(data: dict) -> dict[str, np.ndarray]:
    """从逐步数据计算滚动在线指标时序。"""
    n = len(data["t"])
    result = {}

    # 1. 滚动 NIS 超界比例
    nis = data["nis"]
    nis_exceed = (nis > CHI2_NIS_99).astype(float)
    result["nis_frac"] = _rolling_mean(nis_exceed, NIS_WINDOW)

    # 2-4. 线性化差距（窗口均值）
    for key in ["lin_gap_mean", "lin_gap_cov", "lin_gap_az", "lin_gap_el"]:
        gap = data.get(key)
        if gap is not None:
            result[key] = _rolling_mean(gap, LIN_GAP_WINDOW)

    # 5-6. P 条件数
    p_cond = data.get("P_cond")
    p_eig_min = data.get("P_eig_min")
    if p_cond is not None:
        result["log10_P_cond"] = _rolling_mean(np.log10(np.maximum(p_cond, 1e-300)), P_COND_WINDOW)
    if p_eig_min is not None:
        result["neg_log10_P_eig_min"] = _rolling_mean(
            -np.log10(np.maximum(p_eig_min, 1e-300)), P_COND_WINDOW)

    return result


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """后向滚动均值，前半部分用累积均值填充。"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = np.nanmean(x[start:i + 1])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 阈值搜索与评估
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorEval:
    name: str
    threshold: float
    direction: str        # "above" or "below"
    train_tpr: float
    train_fpr: float
    test_tpr: float
    test_fpr: float
    test_median_lead_time: float | None  # 步数，正=提前
    test_tp_count: int
    test_fn_count: int
    test_fp_count: int
    test_tn_count: int
    test_total_mismatch: int
    test_total_no_mismatch: int
    passed: bool  # TPR>=0.8, FPR<=0.1, lead_time>0
    predictor_or_detector: str  # "predictor", "detector", "fail"


def evaluate_per_trajectory(indicators_list: list[np.ndarray],
                             labels_list: list[TrajectoryLabels],
                             threshold: float, direction: str = "above",
                             ) -> tuple[float, float, float | None, int, int, int, int]:
    """在多个轨迹上评估单个指标在给定阈值下的性能。

    Returns: (tpr, fpr, median_lead_time, tp, fn, fp, tn)
    Event-level: per-trajectory.
    """
    tp, fn, fp, tn = 0, 0, 0, 0
    lead_times = []

    for ind_ts, label in zip(indicators_list, labels_list):
        if direction == "above":
            alarm_mask = ind_ts > threshold
        else:
            alarm_mask = ind_ts < threshold
        alarm_steps = np.where(alarm_mask)[0]
        first_alarm = int(alarm_steps[0]) if len(alarm_steps) > 0 else None

        if label.has_mismatch:
            if first_alarm is not None and first_alarm <= label.onset_step:
                tp += 1
                lead_times.append(label.onset_step - first_alarm)
            else:
                fn += 1
        else:
            if first_alarm is not None:
                fp += 1
            else:
                tn += 1

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    median_lead = float(np.median(lead_times)) if lead_times else None

    return tpr, fpr, median_lead, tp, fn, fp, tn


def search_threshold(name: str, indicators_list: list[np.ndarray],
                      labels_list: list[TrajectoryLabels],
                      direction: str = "above",
                      target_fpr: float = 0.10,
                      n_candidates: int = 200,
                      ) -> tuple[float, float, float, float | None]:
    """在训练集上搜索最佳阈值。

    策略：找出 FPR <= target_fpr 的所有阈值，选 TPR 最高者。
    """
    all_vals = np.concatenate([ind[~np.isnan(ind)] for ind in indicators_list])
    if len(all_vals) == 0:
        return float("nan"), 0.0, 1.0, None

    if direction == "above":
        candidates = np.linspace(np.percentile(all_vals, 5),
                                  np.percentile(all_vals, 99), n_candidates)
    else:
        candidates = np.linspace(np.percentile(all_vals, 1),
                                  np.percentile(all_vals, 95), n_candidates)

    best_thr = float("nan")
    best_tpr = 0.0
    best_fpr = 1.0
    best_lead = None

    for thr in candidates:
        tpr, fpr, lead, _, _, _, _ = evaluate_per_trajectory(
            indicators_list, labels_list, thr, direction)
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_fpr = fpr
            best_thr = thr
            best_lead = lead

    return best_thr, best_tpr, best_fpr, best_lead
        # We need to actually evaluate per-trajectory, but indicators_list has per-trajectory arrays
        # This function iterates differently. Let me restructure.

    # Actually the search needs per-trajectory indicator arrays. Let me fix the loop.
    tpr_list, fpr_list, lead_list = [], [], []

    for thr in candidates:
        tp_all, fn_all, fp_all, tn_all = 0, 0, 0, 0
        lead_times_all = []
        for ind_ts, label in zip(indicators_list, labels_list):
            if direction == "above":
                alarm_mask = ind_ts > thr
            else:
                alarm_mask = ind_ts < thr
            alarm_steps = np.where(alarm_mask)[0]
            first_alarm = int(alarm_steps[0]) if len(alarm_steps) > 0 else None

            if label.has_mismatch:
                if first_alarm is not None and first_alarm <= label.onset_step:
                    tp_all += 1
                    lead_times_all.append(label.onset_step - first_alarm)
                else:
                    fn_all += 1
            else:
                if first_alarm is not None:
                    fp_all += 1
                else:
                    tn_all += 1

        tpr = tp_all / max(tp_all + fn_all, 1)
        fpr = fp_all / max(fp_all + tn_all, 1)
        lead = float(np.median(lead_times_all)) if lead_times_all else None

        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_fpr = fpr
            best_thr = thr
            best_lead = lead

    return best_thr, best_tpr, best_fpr, best_lead


# ═══════════════════════════════════════════════════════════════════════════════
# 主分析入口
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_e2c(data_dir: Path):
    atoms_dir = data_dir / "_atoms"
    if not atoms_dir.exists():
        print(f"Data directory {atoms_dir} not found")
        sys.exit(1)

    # ── 加载所有逐步数据 ──────────────────────────────────────
    print("Loading per-step data...")
    trajectories = []
    for npz_path in sorted(atoms_dir.glob("*.npz")):
        data = dict(np.load(npz_path, allow_pickle=True))
        # Extract metadata from filename
        fname = npz_path.stem  # e.g. "crosstrack_seed1_sigma0.001"
        parts = fname.split("_")
        geom = parts[0]
        seed = int(parts[1].replace("seed", ""))
        sigma_deg = float(parts[2].replace("sigma", ""))
        trajectories.append({
            "geom": geom, "seed": seed, "sigma_deg": sigma_deg,
            "data": data, "path": str(npz_path),
        })

    print(f"Loaded {len(trajectories)} trajectories")

    # ── 计算标签 ──────────────────────────────────────────────
    all_labels: list[TrajectoryLabels] = []
    for traj in trajectories:
        t_arr = traj["data"]["t"]
        nees_arr = traj["data"]["nees"]
        label = compute_labels(t_arr, nees_arr)
        label.geom = traj["geom"]
        label.seed = traj["seed"]
        label.sigma_deg = traj["sigma_deg"]
        all_labels.append(label)

    # ── 计算滚动指标 ──────────────────────────────────────────
    all_indicators = [compute_rolling_indicators(traj["data"]) for traj in trajectories]

    # ── 训练/验证集划分 ──────────────────────────────────────
    train_labels = [l for l in all_labels if l.seed <= 4]
    test_labels = [l for l in all_labels if l.seed >= 5]
    train_indicators = [all_indicators[i] for i, l in enumerate(all_labels) if l.seed <= 4]
    test_indicators = [all_indicators[i] for i, l in enumerate(all_labels) if l.seed >= 5]

    train_mismatch = sum(1 for l in train_labels if l.has_mismatch)
    train_no_mis = sum(1 for l in train_labels if not l.has_mismatch)
    test_mismatch = sum(1 for l in test_labels if l.has_mismatch)
    test_no_mis = sum(1 for l in test_labels if not l.has_mismatch)

    print(f"\nTrain: {len(train_labels)} trajectories, "
          f"mismatch={train_mismatch}, no_mismatch={train_no_mis}")
    print(f"Test:  {len(test_labels)} trajectories, "
          f"mismatch={test_mismatch}, no_mismatch={test_no_mis}")

    # ── 标签分布 ──────────────────────────────────────────────
    print("\n=== Label Distribution ===")
    for geom in GEOMETRY_LABELS:
        for sigma in SIGMA_VALUES:
            for split_name, labels_subset in [("train", train_labels), ("test", test_labels)]:
                group = [l for l in labels_subset
                         if l.geom == geom and abs(l.sigma_deg - sigma) < 0.0005]
                n_mis = sum(1 for l in group if l.has_mismatch)
                if group:
                    print(f"  {geom} sigma={sigma:.3f} {split_name}: "
                          f"mismatch={n_mis}/{len(group)} "
                          f"onset_steps={[l.onset_step for l in group if l.has_mismatch]}")

    # ── 逐指标阈值搜索与评估 ──────────────────────────────────
    print("\n" + "=" * 80)
    print("INDICATOR EVALUATION")
    print("=" * 80)

    indicator_directions = {
        "nis_frac": "above",
        "lin_gap_mean": "above",
        "lin_gap_cov": "above",
        "lin_gap_az": "above",
        "lin_gap_el": "above",
        "log10_P_cond": "above",
        "neg_log10_P_eig_min": "above",
    }

    results = []
    for ind_name in INDICATOR_NAMES:
        direction = indicator_directions.get(ind_name, "above")
        train_ind_arrays = [ind[ind_name] for ind in train_indicators]
        test_ind_arrays = [ind[ind_name] for ind in test_indicators]

        # Train: search threshold
        thr, train_tpr, train_fpr, train_lead = search_threshold(
            ind_name, train_ind_arrays, train_labels, direction)

        if np.isnan(thr):
            results.append(IndicatorEval(
                name=ind_name, threshold=float("nan"), direction=direction,
                train_tpr=0, train_fpr=1, test_tpr=0, test_fpr=1,
                test_median_lead_time=None,
                test_tp_count=0, test_fn_count=test_mismatch,
                test_fp_count=0, test_tn_count=test_no_mis,
                test_total_mismatch=test_mismatch,
                test_total_no_mismatch=test_no_mis,
                passed=False, predictor_or_detector="fail",
            ))
            continue

        # Test: evaluate at chosen threshold
        test_tpr, test_fpr, test_lead, tp_all, fn_all, fp_all, tn_all = \
            evaluate_per_trajectory(test_ind_arrays, test_labels, thr, direction)

        passed = (test_tpr >= 0.8 and test_fpr <= 0.1
                  and test_lead is not None and test_lead > 0)
        if passed:
            ptype = "predictor"
        elif test_tpr >= 0.8 and test_fpr <= 0.1:
            ptype = "detector (lead <= 0)"
        else:
            ptype = "fail"

        eval_result = IndicatorEval(
            name=ind_name, threshold=thr, direction=direction,
            train_tpr=train_tpr, train_fpr=train_fpr,
            test_tpr=test_tpr, test_fpr=test_fpr,
            test_median_lead_time=test_lead,
            test_tp_count=tp_all, test_fn_count=fn_all,
            test_fp_count=fp_all, test_tn_count=tn_all,
            test_total_mismatch=test_mismatch,
            test_total_no_mismatch=test_no_mis,
            passed=passed, predictor_or_detector=ptype,
        )
        results.append(eval_result)

    # ── 打印结果 ──────────────────────────────────────────────
    print(f"\n{'Indicator':<22s} {'Thr':>10s} {'Dir':>6s} "
          f"{'Train TPR':>10s} {'Train FPR':>10s} "
          f"{'Test TPR':>9s} {'Test FPR':>9s} {'Lead(steps)':>12s} {'Verdict':>12s}")
    print("-" * 110)
    for r in results:
        lead_str = f"{r.test_median_lead_time:.0f}" if r.test_median_lead_time is not None else "N/A"
        print(f"{r.name:<22s} {r.threshold:10.4f} {r.direction:>6s} "
              f"{r.train_tpr:10.3f} {r.train_fpr:10.3f} "
              f"{r.test_tpr:9.3f} {r.test_fpr:9.3f} {lead_str:>12s} "
              f"{r.predictor_or_detector:>12s}")

    # ── 详细统计 ──────────────────────────────────────────────
    print("\n=== DETAILS ===")
    for r in results:
        print(f"\n{r.name}: thr={r.threshold:.4f} ({r.direction})")
        print(f"  Train: TPR={r.train_tpr:.3f}, FPR={r.train_fpr:.3f}")
        print(f"  Test:  TPR={r.test_tpr:.3f} ({r.test_tp_count}/{r.test_total_mismatch}), "
              f"FPR={r.test_fpr:.3f} ({r.test_fp_count}/{r.test_total_no_mismatch})")
        print(f"  Lead: {r.test_median_lead_time} steps")
        print(f"  Pass: {r.passed} ({r.predictor_or_detector})")

    # ── 逐案例汇总 ────────────────────────────────────────────
    case_summary = []
    for traj, label, indicators in zip(trajectories, all_labels, all_indicators):
        case = {
            "geom": traj["geom"], "seed": traj["seed"],
            "sigma_deg": traj["sigma_deg"],
            "has_mismatch": label.has_mismatch,
            "onset_step": label.onset_step,
            "onset_time_s": label.onset_time_s,
            "peak_nees": label.peak_nees,
        }
        for r in results:
            ind_ts = indicators.get(r.name)
            if ind_ts is None:
                continue
            if r.direction == "above":
                alarm_mask = ind_ts > r.threshold
            else:
                alarm_mask = ind_ts < r.threshold
            alarm_steps = np.where(alarm_mask)[0]
            first_alarm = int(alarm_steps[0]) if len(alarm_steps) > 0 else None
            case[f"{r.name}_first_alarm"] = first_alarm
            if label.has_mismatch and first_alarm is not None:
                case[f"{r.name}_lead_time"] = (label.onset_step - first_alarm
                                                if label.onset_step is not None else None)
        case_summary.append(case)

    # ── 保存输出 ──────────────────────────────────────────────
    import csv
    with open(data_dir / "indicator_eval.json", "w") as f:
        json.dump([{
            "name": r.name, "threshold": r.threshold, "direction": r.direction,
            "train_tpr": r.train_tpr, "train_fpr": r.train_fpr,
            "test_tpr": r.test_tpr, "test_fpr": r.test_fpr,
            "test_median_lead_time": r.test_median_lead_time,
            "test_tp": r.test_tp_count, "test_fn": r.test_fn_count,
            "test_fp": r.test_fp_count, "test_tn": r.test_tn_count,
            "passed": r.passed, "type": r.predictor_or_detector,
        } for r in results], f, indent=2)

    if case_summary:
        fieldnames = ["geom", "seed", "sigma_deg", "has_mismatch",
                       "onset_step", "onset_time_s", "peak_nees"]
        for r in results:
            fieldnames.append(f"{r.name}_first_alarm")
            fieldnames.append(f"{r.name}_lead_time")
        with open(data_dir / "case_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for cs in case_summary:
                writer.writerow(cs)

    print(f"\nResults saved to {data_dir}/indicator_eval.json")
    print(f"Case summary saved to {data_dir}/case_summary.csv")

    # ── 最终结论 ──────────────────────────────────────────────
    any_predictor = any(r.passed for r in results)
    any_detector = any(r.predictor_or_detector == "detector (lead <= 0)" for r in results)
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if any_predictor:
        print("At least one indicator PASSED as predictor (TPR>=0.8, FPR<=0.1, lead>0).")
    elif any_detector:
        print("No predictor. At least one indicator works as detector (TPR>=0.8, FPR<=0.1, lead<=0).")
    else:
        print("NO indicator passed. Recommended: terminate this method line.")
    print(f"  {sum(1 for r in results if r.passed)}/{len(results)} passed as predictor")

    return results, case_summary


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = DATA_DIR / "e2c_indicator_gate"
    analyze_e2c(data_dir)
