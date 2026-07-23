# ============================================================
# Input / Output Configuration
# ============================================================

PRED_DIR   = "/path/to/predictions"
GT_DIR     = "/path/to/ground_truth/relabeled"
OUTPUT_DIR = "/path/to/output"

X_THRESHOLDS = ["1px", 0.1, 0.2]
MIN_LESION_VOL_MM3 = 2.0

# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from surface_distance import compute_surface_distances, compute_robust_hausdorff
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(OUTPUT_DIR)


def compute_voxel_level_metrics(pred, gt):
    pred_pos = pred > 0
    gt_pos   = gt > 0
    tp = np.sum(pred_pos & gt_pos)
    fp = np.sum(pred_pos & ~gt_pos)
    fn = np.sum(~pred_pos & gt_pos)
    tn = np.sum(~pred_pos & ~gt_pos)
    dsc       = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn)              if (tp + fp + fn) > 0       else 0.0
    recall    = tp / (tp + fn)                   if (tp + fn) > 0             else 0.0
    precision = tp / (tp + fp)                   if (tp + fp) > 0             else 0.0
    return dict(dsc=float(dsc), iou=float(iou), recall=float(recall),
                precision=float(precision),
                voxel_tp=int(tp), voxel_fp=int(fp), voxel_fn=int(fn), voxel_tn=int(tn))


def _spacing_zyx(spacing_xyz):
    return tuple(reversed(spacing_xyz))


def compute_assd(pred, gt, spacing_xyz):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')
    sp = _spacing_zyx(spacing_xyz)
    pb = pred ^ ndimage.binary_erosion(pred)
    gb = gt   ^ ndimage.binary_erosion(gt)
    dt_pred = ndimage.distance_transform_edt(~pb, sampling=sp)
    dt_gt   = ndimage.distance_transform_edt(~gb, sampling=sp)
    d1 = dt_gt[pb]
    d2 = dt_pred[gb]
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    return float((d1.mean() + d2.mean()) / 2.0)


def compute_surface_dice(pred, gt, spacing_xyz, tol_mm=1.0):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp = _spacing_zyx(spacing_xyz)
    pb = pred ^ ndimage.binary_erosion(pred)
    gb = gt   ^ ndimage.binary_erosion(gt)
    dt_pred = ndimage.distance_transform_edt(~pb, sampling=sp)
    dt_gt   = ndimage.distance_transform_edt(~gb, sampling=sp)
    tp1   = np.sum(pb * (dt_gt   <= tol_mm))
    tp2   = np.sum(gb * (dt_pred <= tol_mm))
    denom = np.sum(pb) + np.sum(gb)
    return float((tp1 + tp2) / denom) if denom > 0 else 1.0


def compute_hd95(pred, gt, spacing_xyz):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')
    sp = _spacing_zyx(spacing_xyz)
    sd = compute_surface_distances(pred, gt, sp)
    return float(compute_robust_hausdorff(sd, 95))


def compute_nsd(pred, gt, spacing_xyz, tau=1.0):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp = _spacing_zyx(spacing_xyz)
    sd = compute_surface_distances(pred, gt, sp)
    d_p2g = sd["distances_gt_to_pred"]
    d_g2p = sd["distances_pred_to_gt"]
    p_within = np.sum(d_p2g <= tau) / len(d_p2g) if len(d_p2g) > 0 else 0.0
    g_within = np.sum(d_g2p <= tau) / len(d_g2p) if len(d_g2p) > 0 else 0.0
    return float((p_within + g_within) / 2.0)


def compute_volume_metrics(pred, gt, spacing_xyz):
    vox = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]
    pred_vol = float(np.sum(pred > 0) * vox)
    gt_vol   = float(np.sum(gt   > 0) * vox)
    abs_err  = abs(pred_vol - gt_vol)
    rel_err  = 100.0 * abs_err / gt_vol if gt_vol > 0 else (0.0 if pred_vol == 0 else float('inf'))
    return dict(pred_volume_mm3=pred_vol, gt_volume_mm3=gt_vol,
                volume_abs_error_mm3=abs_err, volume_rel_error_percent=rel_err)


def _get_valid_components(arr, voxel_vol_mm3):
    labeled, n = ndimage.label(arr)
    if n == 0:
        return labeled, np.array([], dtype=int)
    min_vox = max(1, int(np.ceil(MIN_LESION_VOL_MM3 / voxel_vol_mm3)))
    sizes   = np.bincount(labeled.ravel())
    valid   = np.where(sizes[1:] >= min_vox)[0] + 1
    return labeled, valid


def compute_lesion_metrics_two_stage(pred, gt, x_thresh, voxel_vol_mm3=1.0):
    threshold = 0.0 if x_thresh == "1px" else float(x_thresh)

    pred_lab, pred_valid = _get_valid_components(pred > 0, voxel_vol_mm3)
    gt_lab,   gt_valid   = _get_valid_components(gt   > 0, voxel_vol_mm3)

    n_pred = len(pred_valid)
    n_gt   = len(gt_valid)

    gt_mask   = np.isin(gt_lab,   gt_valid)
    pred_mask = np.isin(pred_lab, pred_valid)

    max_pred = int(pred_lab.max()) + 1 if n_pred else 1
    max_gt   = int(gt_lab.max())   + 1 if n_gt   else 1

    pred_sizes      = np.bincount(pred_lab.ravel(), minlength=max_pred)
    pred_overlap_gt = np.bincount(pred_lab[pred_mask & gt_mask].ravel(), minlength=max_pred)

    tp_pred_set = set()
    for p in pred_valid:
        if threshold == 0.0:
            if pred_overlap_gt[p] >= 1:
                tp_pred_set.add(int(p))
        else:
            if pred_overlap_gt[p] / pred_sizes[p] >= threshold:
                tp_pred_set.add(int(p))

    pred_tp_count = len(tp_pred_set)
    pred_fp_count = n_pred - pred_tp_count

    tp_pred_mask = np.isin(pred_lab, list(tp_pred_set)) if tp_pred_set else np.zeros_like(pred_lab, dtype=bool)

    gt_sizes         = np.bincount(gt_lab.ravel(), minlength=max_gt)
    gt_covered_by_tp = np.bincount(gt_lab[gt_mask & tp_pred_mask].ravel(), minlength=max_gt)

    gt_tp_count = 0
    gt_fn_count = 0
    for g in gt_valid:
        if threshold == 0.0:
            detected = gt_covered_by_tp[g] >= 1
        else:
            detected = gt_covered_by_tp[g] / gt_sizes[g] >= threshold
        if detected:
            gt_tp_count += 1
        else:
            gt_fn_count += 1

    lesion_recall    = gt_tp_count / n_gt if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    lesion_precision = pred_tp_count / (pred_tp_count + pred_fp_count) if (pred_tp_count + pred_fp_count) > 0 else (1.0 if n_gt == 0 else 0.0)
    lesion_f1 = 2 * lesion_recall * lesion_precision / (lesion_recall + lesion_precision) if (lesion_recall + lesion_precision) > 0 else 0.0

    x_label = "1px" if x_thresh == "1px" else str(x_thresh)
    return {
        f"lesion_num_gt":               int(n_gt),
        f"lesion_num_pred":             int(n_pred),
        f"lesion_pred_tp_x{x_label}":  int(pred_tp_count),
        f"lesion_fp_x{x_label}":       int(pred_fp_count),
        f"lesion_gt_tp_x{x_label}":    int(gt_tp_count),
        f"lesion_fn_x{x_label}":       int(gt_fn_count),
        f"lesion_recall_x{x_label}":   float(lesion_recall),
        f"lesion_precision_x{x_label}":float(lesion_precision),
        f"lesion_f1_x{x_label}":       float(lesion_f1),
    }


def compute_case_metrics(pred_file, gt_file):
    pred_img = sitk.ReadImage(str(pred_file))
    gt_img   = sitk.ReadImage(str(gt_file))

    pred_b = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
    gt_b   = (sitk.GetArrayFromImage(gt_img)   > 0).astype(np.uint8)

    spacing = list(pred_img.GetSpacing())

    metrics = {}
    metrics.update(compute_voxel_level_metrics(pred_b, gt_b))
    metrics.update(dict(
        assd         = compute_assd(pred_b, gt_b, spacing),
        surface_dice = compute_surface_dice(pred_b, gt_b, spacing),
        hd95         = compute_hd95(pred_b, gt_b, spacing),
        nsd          = compute_nsd(pred_b, gt_b, spacing),
    ))
    metrics.update(compute_volume_metrics(pred_b, gt_b, spacing))

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    for x in X_THRESHOLDS:
        metrics.update(compute_lesion_metrics_two_stage(pred_b, gt_b, x, voxel_vol))

    return metrics


def summarize(df):
    def nanmean(s): return float(np.nanmean(s.replace([np.inf, -np.inf], np.nan)))
    def nanstd(s):  return float(np.nanstd(s.replace([np.inf, -np.inf], np.nan)))

    summary = {}
    for col in ["dsc", "iou", "recall", "precision",
                "assd", "surface_dice", "hd95", "nsd",
                "pred_volume_mm3", "gt_volume_mm3",
                "volume_abs_error_mm3", "volume_rel_error_percent"]:
        if col in df.columns:
            summary[f"{col}_mean"] = nanmean(df[col])
            summary[f"{col}_std"]  = nanstd(df[col])

    for x in X_THRESHOLDS:
        x_label = "1px" if x == "1px" else str(x)
        for metric in ["lesion_recall", "lesion_precision", "lesion_f1"]:
            col = f"{metric}_x{x_label}"
            if col in df.columns:
                summary[f"{col}_mean"] = nanmean(df[col])
                summary[f"{col}_std"]  = nanstd(df[col])
        for count_col in ["lesion_pred_tp", "lesion_gt_tp", "lesion_fp", "lesion_fn"]:
            col = f"{count_col}_x{x_label}"
            if col in df.columns:
                summary[f"{col}_total"] = int(df[col].sum())
    if "lesion_num_gt"   in df.columns: summary["lesion_num_gt_total"]   = int(df["lesion_num_gt"].sum())
    if "lesion_num_pred" in df.columns: summary["lesion_num_pred_total"] = int(df["lesion_num_pred"].sum())
    summary["n_cases"] = len(df)
    return summary


def print_report(summary):
    print(f"\n{'='*70}")
    print(f"  n={summary['n_cases']}")
    print(f"{'='*70}")
    print(f"  [Voxel]  DSC={summary['dsc_mean']:.4f}±{summary['dsc_std']:.4f}  "
          f"IoU={summary['iou_mean']:.4f}±{summary['iou_std']:.4f}  "
          f"Recall={summary['recall_mean']:.4f}  Precision={summary['precision_mean']:.4f}")
    print(f"  [Boundary]  ASSD={summary['assd_mean']:.3f}mm  "
          f"Surface_DSC={summary['surface_dice_mean']:.4f}  "
          f"HD95={summary['hd95_mean']:.3f}mm  NSD={summary['nsd_mean']:.4f}")
    for x in X_THRESHOLDS:
        x_label = "1px" if x == "1px" else str(x)
        rec  = summary.get(f"lesion_recall_x{x_label}_mean", float('nan'))
        prec = summary.get(f"lesion_precision_x{x_label}_mean", float('nan'))
        f1   = summary.get(f"lesion_f1_x{x_label}_mean", float('nan'))
        pred_tp = summary.get(f"lesion_pred_tp_x{x_label}_total", "?")
        gt_tp   = summary.get(f"lesion_gt_tp_x{x_label}_total", "?")
        fp      = summary.get(f"lesion_fp_x{x_label}_total", "?")
        fn      = summary.get(f"lesion_fn_x{x_label}_total", "?")
        print(f"  [Lesion X={x_label}]  Recall={rec:.4f}  Precision={prec:.4f}  F1={f1:.4f}  "
              f"pred_TP={pred_tp}  gt_TP={gt_tp}  FP={fp}  FN={fn}")
    print(f"  [Volume]  AbsErr={summary['volume_abs_error_mm3_mean']:.1f}±{summary['volume_abs_error_mm3_std']:.1f}mm³  "
          f"RelErr={summary['volume_rel_error_percent_mean']:.2f}%")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_files = {f.name: f for f in Path(PRED_DIR).glob("*.nii.gz")}
    gt_files   = {f.name: f for f in Path(GT_DIR).glob("*.nii.gz")}
    common = sorted(set(pred_files) & set(gt_files))

    if not common:
        print(f"[ERROR] No matching .nii.gz files found.\n  PRED: {PRED_DIR}\n  GT:   {GT_DIR}")
        return

    print(f"Found {len(common)} matched cases (pred={len(pred_files)}, gt={len(gt_files)})")

    rows = []
    for idx, name in enumerate(common, 1):
        try:
            m = compute_case_metrics(pred_files[name], gt_files[name])
            m["case"] = name
            rows.append(m)
            if idx % 10 == 0 or idx == len(common):
                print(f"  [{idx}/{len(common)}] done", flush=True)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "case_metrics.csv", index=False)

    summary = summarize(df)
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    print_report(summary)
    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
