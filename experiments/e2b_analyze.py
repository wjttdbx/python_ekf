"""
E2b 配对比较分析：Flow-Jacobian vs Flow-UKF

读取 UKF 结果和 E2 Jacobian 基线，按 (geometry, seed, sigma_theta_deg) 配对合并，
计算配对差、bootstrap CI、边界指标。
"""

from __future__ import annotations
import csv, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import chi2

warnings.filterwarnings("ignore")

from aerospace.paths import DATA_DIR

SIGMA_THETA_VALUES = [0.001, 0.008, 0.1]
GEOMETRY_LABELS = ["radial", "alongtrack", "crosstrack"]

# ═══════════════════════════════════════════════════════════════════════════════
# 指标字典
# ═══════════════════════════════════════════════════════════════════════════════

NUMERIC_METRICS = [
    "T_FP", "v_FP", "T_soft", "T_sustain",
    "total_delta_v", "peak_accel", "control_energy",
    "rmse_pos_x", "rmse_pos_y", "rmse_pos_z", "rmse_dist",
    "final_pos_err", "final_vel_err",
    "est_dist_bias", "est_radial_vel_bias",
    "nis_mean", "nees_mean", "nis_inlier_frac", "nees_inlier_frac",
    "are_fallback_count", "exit_count", "time_in_sphere_frac",
    "P_eig_min", "P_cond",
]

BOOL_METRICS = ["sustained"]


def _to_float(val) -> float | None:
    if val is None or val == "" or val == "None":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


# ═══════════════════════════════════════════════════════════════════════════════
# 加载与配对
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: Path, nav_label: str) -> dict[tuple, dict]:
    """返回 {(geometry, seed, sigma): row_dict}。"""
    data = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["geometry"], int(row["seed"]), float(row["sigma_theta_deg"]))
            row["_nav"] = nav_label
            data[key] = row
    return data


def merge_pairs(jacobian_data: dict, ukf_data: dict) -> list[dict]:
    """返回配对行列表，每行包含 jacobian_* 和 ukf_* 前缀的指标及差值。"""
    pairs = []
    common_keys = set(jacobian_data.keys()) & set(ukf_data.keys())
    missing_j = set(ukf_data.keys()) - set(jacobian_data.keys())
    missing_u = set(jacobian_data.keys()) - set(ukf_data.keys())
    if missing_j:
        print(f"Warning: {len(missing_j)} keys only in UKF, missing from Jacobian")
    if missing_u:
        print(f"Warning: {len(missing_u)} keys only in Jacobian, missing from UKF")

    for key in sorted(common_keys):
        j_row = jacobian_data[key]
        u_row = ukf_data[key]
        pair = {
            "geometry": key[0], "seed": key[1], "sigma_theta_deg": key[2],
        }
        for metric in NUMERIC_METRICS:
            j_val = _to_float(j_row.get(metric))
            u_val = _to_float(u_row.get(metric))
            pair[f"jacobian_{metric}"] = j_val
            pair[f"ukf_{metric}"] = u_val
            if j_val is not None and u_val is not None:
                pair[f"delta_{metric}"] = u_val - j_val  # UKF - Jacobian
            else:
                pair[f"delta_{metric}"] = None
        for metric in BOOL_METRICS:
            pair[f"jacobian_{metric}"] = _to_bool(j_row.get(metric, False))
            pair[f"ukf_{metric}"] = _to_bool(u_row.get(metric, False))
        pairs.append(pair)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(deltas: list[float], n_bootstrap: int = 10000,
                 alpha: float = 0.05, rng_seed: int = 42) -> dict:
    """Bootstrap 95% CI for the mean of paired differences."""
    deltas = np.array([d for d in deltas if d is not None and not np.isnan(d)])
    n = len(deltas)
    if n < 3:
        return {"mean": float(np.nan), "ci_lower": float(np.nan),
                "ci_upper": float(np.nan), "n_valid": n}
    rng = np.random.default_rng(rng_seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=n, replace=True)
        means.append(float(np.mean(sample)))
    means = np.sort(means)
    ci_lower = float(means[int(alpha / 2 * n_bootstrap)])
    ci_upper = float(means[int((1 - alpha / 2) * n_bootstrap)])
    return {"mean": float(np.mean(deltas)), "ci_lower": ci_lower,
            "ci_upper": ci_upper, "n_valid": n}


# ═══════════════════════════════════════════════════════════════════════════════
# 主分析
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(e2b_dir: Path, e2_csv: Path):
    ukf_csv = e2b_dir / "ukf_results.csv"
    if not ukf_csv.exists():
        print(f"UKF results not found at {ukf_csv}")
        sys.exit(1)

    jacobian = load_csv(e2_csv, "jacobian")
    ukf = load_csv(ukf_csv, "ukf")
    pairs = merge_pairs(jacobian, ukf)
    print(f"Paired cases: {len(pairs)}")

    # ── Per-condition summary ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-CONDITION SUMMARY: NEES mean / NIS mean / NIS inlier frac")
    print("=" * 80)
    print(f"{'geom':>12s} {'sigma':>7s} {'Jac NEES':>10s} {'UKF NEES':>10s} "
          f"{'Jac NIS':>8s} {'UKF NIS':>8s} "
          f"{'Jac inlier':>10s} {'UKF inlier':>10s}")

    condition_summaries = []
    for geom in GEOMETRY_LABELS:
        for sigma in SIGMA_THETA_VALUES:
            group = [p for p in pairs
                     if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            j_nees = [_to_float(p["jacobian_nees_mean"]) for p in pairs
                      if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            u_nees = [_to_float(p["ukf_nees_mean"]) for p in pairs
                      if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            j_nis = [_to_float(p["jacobian_nis_mean"]) for p in pairs
                     if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            u_nis = [_to_float(p["ukf_nis_mean"]) for p in pairs
                     if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            j_inlier = [p.get("jacobian_nis_inlier_frac") for p in group]
            u_inlier = [p.get("ukf_nis_inlier_frac") for p in group]
            # Jacobian E2 data doesn't have nis_inlier_frac, so j_inlier may be None

            j_nees_m = np.mean([v for v in j_nees if v is not None]) if j_nees else float("nan")
            u_nees_m = np.mean([v for v in u_nees if v is not None]) if u_nees else float("nan")
            j_nis_m = np.mean([v for v in j_nis if v is not None]) if j_nis else float("nan")
            u_nis_m = np.mean([v for v in u_nis if v is not None]) if u_nis else float("nan")
            j_inl = np.mean([v for v in j_inlier if v is not None]) if any(v is not None for v in j_inlier) else float("nan")
            u_inl = np.mean([v for v in u_inlier if v is not None]) if any(v is not None for v in u_inlier) else float("nan")

            print(f"{geom:>12s} {sigma:7.3f} {j_nees_m:10.1f} {u_nees_m:10.1f} "
                  f"{j_nis_m:8.3f} {u_nis_m:8.3f} "
                  f"{j_inl:10.3f} {u_inl:10.3f}")

            condition_summaries.append({
                "geometry": geom, "sigma_theta_deg": sigma,
                "jacobian_nees_mean": j_nees_m, "ukf_nees_mean": u_nees_m,
                "jacobian_nis_mean": j_nis_m, "ukf_nis_mean": u_nis_m,
                "jacobian_nis_inlier_frac": j_inl if not np.isnan(j_inl) else None,
                "ukf_nis_inlier_frac": u_inl,
            })

    # ── Paired-difference analysis ──────────────────────────────────
    print("\n" + "=" * 80)
    print("PAIRED DIFFERENCES (UKF - Jacobian) with Bootstrap 95% CI")
    print("=" * 80)

    key_metrics = ["nees_mean", "nis_mean", "rmse_dist", "T_FP",
                   "total_delta_v", "peak_accel"]

    paired_summary = []
    for metric in key_metrics:
        all_deltas = []
        for p in pairs:
            d = p.get(f"delta_{metric}")
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                all_deltas.append(d)
        ci = bootstrap_ci(all_deltas)

        # Per-geometry breakdown
        for geom in GEOMETRY_LABELS:
            geo_deltas = []
            for p in pairs:
                if p["geometry"] != geom:
                    continue
                d = p.get(f"delta_{metric}")
                if d is not None and not (isinstance(d, float) and np.isnan(d)):
                    geo_deltas.append(d)
            geo_ci = bootstrap_ci(geo_deltas)
            print(f"  {metric:>18s} {geom:>12s}: mean={geo_ci['mean']:+.4e} "
                  f"95%CI=[{geo_ci['ci_lower']:+.4e}, {geo_ci['ci_upper']:+.4e}] "
                  f"n={geo_ci['n_valid']}")
        print(f"  {metric:>18s} {'ALL':>12s}: mean={ci['mean']:+.4e} "
              f"95%CI=[{ci['ci_lower']:+.4e}, {ci['ci_upper']:+.4e}] "
              f"n={ci['n_valid']}")
        print()

        paired_summary.append({
            "metric": metric,
            "overall": ci,
        })

    # ── Boundary indicators ─────────────────────────────────────────
    print("=" * 80)
    print("BOUNDARY INDICATORS")
    print("=" * 80)

    for geom in GEOMETRY_LABELS:
        for sigma in SIGMA_THETA_VALUES:
            group = [p for p in pairs
                     if p["geometry"] == geom and abs(p["sigma_theta_deg"] - sigma) < 0.0005]
            if not group:
                continue
            u_nees_inlier = [p.get("ukf_nees_inlier_frac") for p in group
                           if p.get("ukf_nees_inlier_frac") is not None]
            u_nis_inlier = [p.get("ukf_nis_inlier_frac") for p in group
                          if p.get("ukf_nis_inlier_frac") is not None]
            u_P_eig_min = [p.get("ukf_P_eig_min") for p in group
                          if p.get("ukf_P_eig_min") is not None]
            u_P_cond = [p.get("ukf_P_cond") for p in group
                       if p.get("ukf_P_cond") is not None]

            nees_inl = np.mean(u_nees_inlier) if u_nees_inlier else float("nan")
            nis_inl = np.mean(u_nis_inlier) if u_nis_inlier else float("nan")
            pmin = np.mean(u_P_eig_min) if u_P_eig_min else float("nan")
            pcond = np.mean(u_P_cond) if u_P_cond else float("nan")

            print(f"  {geom:>12s} sigma={sigma:.3f}: "
                  f"NEES_inlier={nees_inl:.3f} NIS_inlier={nis_inl:.3f} "
                  f"P_eig_min={pmin:.2e} P_cond={pcond:.1e}")

    # ── Outlier/anomalous seed detection ────────────────────────────
    print("\n" + "=" * 80)
    print("ANOMALOUS SEED DETECTION (NEES > 100 in Jacobian)")
    print("=" * 80)
    anomalous = []
    for p in pairs:
        j_nees = p.get("jacobian_nees_mean")
        if j_nees is not None and j_nees > 100:
            u_nees = p.get("ukf_nees_mean")
            print(f"  {p['geometry']} sigma={p['sigma_theta_deg']} seed={p['seed']}: "
                  f"Jac NEES={j_nees:.1f} → UKF NEES={u_nees:.1f}")
            anomalous.append(p)
    if not anomalous:
        print("  (none found with NEES > 100)")

    # ── Save paired_comparison.csv ──────────────────────────────────
    paired_csv_path = e2b_dir / "paired_comparison.csv"
    if pairs:
        all_fields = ["geometry", "seed", "sigma_theta_deg"]
        for metric in NUMERIC_METRICS:
            all_fields.append(f"jacobian_{metric}")
            all_fields.append(f"ukf_{metric}")
            all_fields.append(f"delta_{metric}")
        for metric in BOOL_METRICS:
            all_fields.append(f"jacobian_{metric}")
            all_fields.append(f"ukf_{metric}")

        with open(paired_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
            writer.writeheader()
            for p in pairs:
                writer.writerow(p)
        print(f"\nPaired comparison saved to {paired_csv_path}")

    # ── Save analysis summary JSON ──────────────────────────────────
    analysis_summary = {
        "experiment": "E2b paired analysis",
        "total_pairs": len(pairs),
        "condition_summaries": condition_summaries,
        "paired_differences": [
            {"metric": ps["metric"], "mean": ps["overall"]["mean"],
             "ci_lower": ps["overall"]["ci_lower"],
             "ci_upper": ps["overall"]["ci_upper"],
             "n": ps["overall"]["n_valid"]}
            for ps in paired_summary
        ],
        "anomalous_seeds": [
            {"geometry": a["geometry"], "seed": a["seed"],
             "sigma_theta_deg": a["sigma_theta_deg"],
             "jacobian_nees": a.get("jacobian_nees_mean"),
             "ukf_nees": a.get("ukf_nees_mean")}
            for a in anomalous
        ],
    }
    with open(e2b_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"Analysis summary saved to {e2b_dir / 'analysis_summary.json'}")

    return pairs, anomalous


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    e2b_dir = DATA_DIR / "e2b_geometry_filter_boundary"
    e2_csv = DATA_DIR / "e2_flow_isolated_n10" / "e2_results.csv"
    analyze(e2b_dir, e2_csv)
