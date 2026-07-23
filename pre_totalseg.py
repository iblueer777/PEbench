# --------------------------------------------
# Description: 3D nnU-Net preprocessing pipeline for PE detection dataset
# Converts .nii and .nrrd data into nnU-Net raw format
# --------------------------------------------
"""
Integrated data preprocessing pipeline:
1. Read raw .nii and .nrrd data
2. Segment lungs using TotalSegmentator
3. Crop CT and label volumes to the lung region
4. Convert to nnU-Net format

Usage:
python pre_totalsg_inspect_en.py \
    -i <path_to_raw_data> \
    -o <path_to_nnUNet_raw>/DatasetXXX_NAME \
    --margin 10 \
    --dataset_name DatasetXXX_NAME \
    -n 1
"""


import os
import re
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import nrrd
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


# ==================== Utility functions ====================

def _file_ok(p: str) -> bool:
    """Check that a file exists and is non-empty"""
    try:
        return os.path.isfile(p) and os.path.getsize(p) > 0
    except Exception:
        return False


def sanitize(s: str) -> str:
    """Sanitize a name, keeping only alphanumerics and underscores"""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


# ==================== TotalSegmentator ====================

def run_totalsegmentator(input_file, output_dir):
    """
    Run TotalSegmentator to segment the lungs.

    Key point: input file must not be located inside the output directory.
    """
    from totalsegmentator.python_api import totalsegmentator

    print(f"    Running TotalSegmentator...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        totalsegmentator(
            input=str(input_file),
            output=str(output_dir),
            fast=True,
            ml=True,
            quiet=False,
            task="total",
        )

        possible_locations = [
            output_dir,
            output_dir.parent / f"{output_dir.name}.nii",
        ]

        print(f"\n    Checking output locations...")
        actual_output = None

        for check_path in possible_locations:
            print(f"    Checking: {check_path}")
            if check_path.exists():
                if check_path.is_dir():
                    items = list(check_path.iterdir())
                    print(f"      [DIR] item count: {len(items)}")
                    if len(items) > 0:
                        print(f"      Directory contents:")
                        for item in items[:5]:
                            if item.is_file():
                                size_mb = item.stat().st_size / (1024*1024)
                                print(f"        - [FILE] {item.name} ({size_mb:.2f} MB)")
                            else:
                                print(f"        - [DIR]  {item.name}")
                        actual_output = check_path
                        break
                elif check_path.is_file():
                    size_mb = check_path.stat().st_size / (1024*1024)
                    print(f"      [FILE] size: {size_mb:.2f} MB")
                    actual_output = check_path
                    break

        if actual_output is None:
            raise RuntimeError("Could not locate TotalSegmentator output")

        # === Single multi-label output file ===
        if actual_output.is_file():
            print(f"    Processing single multi-label segmentation file: {actual_output.name}")

            seg_img = nib.load(actual_output)
            seg_data = seg_img.get_fdata()

            print(f"    Segmentation shape: {seg_data.shape}")
            print(f"    Label values: {np.unique(seg_data)}")

            lung_labels = list(range(1, 11))
            lung_mask = np.isin(seg_data, lung_labels)

            if not lung_mask.any():
                print(f"    Warning: falling back to all non-zero labels as lung")
                lung_mask = seg_data > 0

            lung_file = output_dir / "lung_combined.nii.gz"
            lung_img = nib.Nifti1Image(
                lung_mask.astype(np.uint8),
                seg_img.affine,
                seg_img.header
            )
            nib.save(lung_img, lung_file)

            actual_output.unlink()

            print(f"    OK saved combined lung mask: {lung_file.name}")
            print(f"      Lung voxel count: {lung_mask.sum()}")

            return True

        # === Directory of per-structure files ===
        else:
            all_nii_gz = list(actual_output.rglob("*.nii.gz"))
            all_nii = list(actual_output.rglob("*.nii"))
            all_files = all_nii_gz + all_nii
            lung_files = [f for f in all_files if 'lung' in f.name.lower()]

            if len(lung_files) == 0:
                raise RuntimeError("No lung lobe segmentation files found")

            print(f"    Found {len(lung_files)} lung lobe files")

            for lung_file in lung_files:
                if lung_file.suffix == '.nii':
                    img = nib.load(lung_file)
                    gz_file = lung_file.with_suffix('.nii.gz')
                    nib.save(img, gz_file)
                    lung_file.unlink()
                    lung_file = gz_file

                target = output_dir / lung_file.name
                if lung_file != target:
                    shutil.move(str(lung_file), str(target))

            if actual_output != output_dir and actual_output.exists():
                shutil.rmtree(actual_output, ignore_errors=True)

            final_files = sorted(output_dir.glob("*lung*.nii.gz"))
            print(f"    OK kept {len(final_files)} lung lobe segmentations")
            for seg_file in final_files:
                voxels = np.sum(nib.load(seg_file).get_fdata() > 0)
                print(f"      - {seg_file.name}: {voxels} voxels")

            return True

    except Exception as e:
        print(f"    FAILED TotalSegmentator: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_lung_bbox(segmentation_dir, margin=10):
    """Compute a bounding box from the lung segmentation results"""
    seg_files = list(Path(segmentation_dir).glob("*.nii.gz"))

    if not seg_files:
        return None, None

    reference_img = nib.load(seg_files[0])
    lung_mask = np.zeros(reference_img.shape, dtype=bool)

    print(f"    Merging {len(seg_files)} lung lobes...")
    for seg_file in seg_files:
        seg_data = nib.load(seg_file).get_fdata()
        lung_mask[seg_data > 0] = True

    coords = np.argwhere(lung_mask > 0)

    if len(coords) == 0:
        return None, None

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    x_min = max(0, min_coords[0] - margin)
    x_max = min(lung_mask.shape[0], max_coords[0] + margin + 1)

    y_min = max(0, min_coords[1] - margin)
    y_max = min(lung_mask.shape[1], max_coords[1] + margin + 1)

    z_min = max(0, min_coords[2] - margin)
    z_max = min(lung_mask.shape[2], max_coords[2] + margin + 1)

    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)

    return bbox, reference_img


def process_single_case(args):
    """
    Process a single case: segment, crop, save.

    Input file and output directory are kept fully separate.
    """
    (input_image, input_seg, output_image, output_seg,
     case_id, patient_id, temp_dir, margin) = args

    if not _file_ok(input_image):
        return ("skip", f"bad_image_file:{input_image}")
    if not _file_ok(input_seg):
        return ("skip", f"bad_mask_file:{input_seg}")

    try:
        # 1. Load raw data
        nii_img = nib.load(input_image)
        img_data = nii_img.get_fdata()

        if img_data.dtype != np.float32:
            img_data = img_data.astype(np.float32)

        affine = nii_img.affine
        header = nii_img.header
        original_shape = img_data.shape

        seg_data, seg_header = nrrd.read(input_seg)
        seg_data = (seg_data > 0).astype(np.uint8)

        if img_data.shape != seg_data.shape:
            print(f"    Warning: CT/label shape mismatch: {img_data.shape} vs {seg_data.shape}")
            if seg_data.shape == img_data.shape[::-1]:
                seg_data = np.transpose(seg_data, (2, 1, 0))
                print(f"    -> transposed label to: {seg_data.shape}")
            else:
                return ("skip", f"shape_mismatch:{img_data.shape}_vs_{seg_data.shape}")

        # 2. Write a temporary CT file (in temp_dir root)
        temp_ct_file = Path(temp_dir) / f"{case_id}.nii.gz"
        nib.save(nii_img, temp_ct_file)

        # 3. Run TotalSegmentator into its own output directory
        seg_output_dir = Path(temp_dir) / f"seg_{case_id}"
        seg_output_dir.mkdir(exist_ok=True)

        print(f"    Processing {case_id}...")
        success = run_totalsegmentator(temp_ct_file, seg_output_dir)

        if not success:
            temp_ct_file.unlink(missing_ok=True)
            shutil.rmtree(seg_output_dir, ignore_errors=True)
            return ("skip", f"totalseg_failed:{case_id}")

        # 4. Compute bounding box
        print("  Computing bounding box...")
        bbox, ref_img = get_lung_bbox(seg_output_dir, margin=margin)

        if bbox is None:
            temp_ct_file.unlink(missing_ok=True)
            shutil.rmtree(seg_output_dir, ignore_errors=True)
            return ("skip", f"no_lung_mask:{case_id}")

        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        print(f"    Bounding box: x[{x_min}:{x_max}] y[{y_min}:{y_max}] z[{z_min}:{z_max}]")

        # 5. Crop
        print("  Cropping CT and label...")
        cropped_img_data = img_data[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        cropped_seg_data = seg_data[x_min:x_max, y_min:y_max, z_min:z_max].copy()

        cropped_shape = cropped_img_data.shape

        new_affine = affine.copy()
        offset = np.array([x_min, y_min, z_min])
        new_affine[:3, 3] = affine[:3, :3] @ offset + affine[:3, 3]

        cropped_img = nib.Nifti1Image(cropped_img_data, new_affine, header)
        cropped_seg = nib.Nifti1Image(cropped_seg_data, new_affine, header)

        # 6. Save
        nib.save(cropped_img, output_image)
        nib.save(cropped_seg, output_seg)
        print(f"  OK saved: {Path(output_image).name}")

        # 7. Clean up
        temp_ct_file.unlink(missing_ok=True)
        shutil.rmtree(seg_output_dir, ignore_errors=True)

        # 8. Stats
        fg_voxels_original = np.sum(seg_data > 0)
        fg_voxels_cropped = np.sum(cropped_seg_data > 0)

        bbox_info = {
            'bbox': [int(x) for x in bbox],
            'original_shape': [int(x) for x in original_shape],
            'cropped_shape': [int(x) for x in cropped_shape],
            'foreground_voxels_original': int(fg_voxels_original),
            'foreground_voxels_cropped': int(fg_voxels_cropped),
            'foreground_ratio': float(fg_voxels_cropped / fg_voxels_original) if fg_voxels_original > 0 else 0.0
        }

        print(f"    OK {case_id}: {original_shape} -> {cropped_shape}")
        print(f"      Foreground voxels: {fg_voxels_original} -> {fg_voxels_cropped} ({bbox_info['foreground_ratio']*100:.1f}% kept)")

        return ("ok", (case_id, patient_id, bbox_info))

    except Exception as e:
        print(f"    FAILED {case_id}: {e}")
        import traceback
        traceback.print_exc()
        return ("skip", f"process_error:{str(e)}")


# ==================== Main pipeline ====================

def prepare_dataset(source_root, output_root, dataset_name, margin=10, n_processes=4):
    """
    Main entry point: integrated data preprocessing pipeline.

    Args:
        source_root: root directory of raw data (may contain arbitrary subfolders)
        output_root: output root directory (nnUNet_raw)
        dataset_name: dataset name
        margin: bounding box expansion margin
        n_processes: number of parallel processes
    """
    print(f"\n{'='*70}")
    print(f"Preparing dataset: {dataset_name}")
    print(f"Source data: {source_root}")
    print(f"Output directory: {output_root}")
    print(f"Bounding box margin: {margin}")
    print(f"{'='*70}\n")

    # Create directory structure
    ds_root = join(output_root, dataset_name)
    imagestr = join(ds_root, "imagesTr")
    labelstr = join(ds_root, "labelsTr")
    temp_dir = join(ds_root, "temp")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(temp_dir)

    # ==================== Check data directory ====================
    if not os.path.isdir(source_root):
        raise RuntimeError(f"source_root not found: {source_root}")

    print(f"Scanning data directory: {source_root}")

    # Supports two input layouts:
    # 1) Each case in its own subfolder containing .nii/.nii.gz and .nrrd
    # 2) All .nii.gz/.nii and .nrrd files mixed in one or more directories
    #
    # We recursively scan source_root and pair image/seg files by basename.
    def _base_name(fn: str) -> str:
        if fn.lower().endswith('.nii.gz'):
            return fn[:-7]
        if fn.lower().endswith('.nii'):
            return fn[:-4]
        if fn.lower().endswith('.nrrd'):
            return fn[:-5]
        return fn

    source_path = Path(source_root)
    all_files = list(source_path.rglob("*"))
    nii_candidates = []
    nrrd_candidates = []

    for p in all_files:
        if not p.is_file():
            continue
        ln = p.name.lower()
        if ln.endswith('.nii') or ln.endswith('.nii.gz'):
            nii_candidates.append(p)
        elif ln.endswith('.nrrd'):
            nrrd_candidates.append(p)

    print(f"  Found {len(nii_candidates)} nii/nii.gz files, {len(nrrd_candidates)} nrrd files (recursive)")

    # Build basename index
    images_by_base = defaultdict(list)
    segs_by_base = defaultdict(list)

    for p in nii_candidates:
        b = _base_name(p.name)
        images_by_base[b].append(p)

    for p in nrrd_candidates:
        b = _base_name(p.name)
        segs_by_base[b].append(p)

    # Pair: basename present in both image and seg sets
    matched_bases = sorted([b for b in images_by_base.keys() if b in segs_by_base])
    print(f"  Matched {len(matched_bases)} paired cases (by basename)\n")

    args = []
    total_pairs = 0

    for idx, base in enumerate(matched_bases, 1):
        # Choose image: prefer .nii.gz, then .nii (first match if multiple)
        imgs = images_by_base[base]
        img_path = None
        for cand in imgs:
            if cand.name.lower().endswith('.nii.gz'):
                img_path = cand
                break
        if img_path is None:
            img_path = imgs[0]

        # Choose seg: first match if multiple .nrrd files
        seg_path = segs_by_base[base][0]

        # patient_id rule:
        # - If image/seg live in a subfolder (not directly under source_root),
        #   use the parent folder name as patient_id.
        # - Otherwise use the basename as patient_id (flat layout default).
        parent = img_path.parent
        if parent != source_path:
            patient_id = sanitize(parent.name)
        else:
            patient_id = sanitize(base)

        case_id = f"pe_{patient_id}_{sanitize(base)}"

        out_img = join(imagestr, case_id + "_0000.nii.gz")
        out_seg = join(labelstr, case_id + ".nii.gz")

        # Skip if target files already exist
        if _file_ok(out_img) and _file_ok(out_seg):
            print(f"[{idx}/{len(matched_bases)}] Skipping existing: {case_id}")
            continue

        args.append((str(img_path), str(seg_path), out_img, out_seg,
                     case_id, patient_id, temp_dir, margin))
        total_pairs += 1
        print(f"[{idx}/{len(matched_bases)}] Paired: {img_path.name} + {seg_path.name} -> case_id: {case_id}")

    # ==================== Summary ====================
    print(f"\n{'='*70}")
    print(f"Data collection complete:")
    print(f"  Scanned files: nii/nii.gz={len(nii_candidates)}, nrrd={len(nrrd_candidates)}")
    print(f"  Successfully paired: {total_pairs}")
    print(f"  Cases to process: {len(args)}")
    print(f"{'='*70}\n")

    if len(args) == 0:
        print("ERROR: no processable cases found!")
        print("\nPossible reasons:")
        print("1. .nii and .nrrd filenames do not match")
        print("2. Files are empty or corrupted")
        print("3. File permission issues")
        return

    # ==================== Test the first sample ====================
    print("Processing the first sample as a test...")
    print(f"Test sample: {args[0][4]}")  # case_id
    print(f"  Image: {args[0][0]}")
    print(f"  Label: {args[0][1]}")

    test_result = process_single_case(args[0])
    print(f"\nTest result: {test_result[0]}")

    if test_result[0] != "ok":
        print(f"ERROR: test sample failed: {test_result[1]}")
        print("\nPlease fix the issue before batch processing")
        print("\nCommon issues:")
        print("1. TotalSegmentator not installed or path issue")
        print("2. Image format issues (spacing, origin, etc.)")
        print("3. Out of memory")
        return

    print("OK test sample processed successfully!")

    # Ask whether to continue
    if len(args) > 1:
        print(f"\nReady to process the remaining {len(args) - 1} cases")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    # ==================== Batch processing ====================
    print(f"\n{'='*70}")
    print(f"Batch processing configuration:")
    print(f"  Processes: {n_processes}")
    print(f"  Remaining cases: {len(args) - 1}")
    print(f"{'='*70}\n")

    if n_processes == 1 or len(args) == 1:
        print("Using single-process sequential processing...\n")
        results = [test_result]
        if len(args) > 1:
            for arg in tqdm(args[1:], desc="Processing cases"):
                result = process_single_case(arg)
                results.append(result)
    else:
        print(f"Using {n_processes} parallel processes...\n")
        with Pool(processes=n_processes) as p:
            results = [test_result] + list(tqdm(
                p.imap(process_single_case, args[1:]),
                total=len(args) - 1,
                desc="Processing cases"
            ))

    # ==================== Result summary ====================
    ok_results = [msg for s, msg in results if s == "ok"]
    skip_reasons = [msg for s, msg in results if s == "skip"]

    # Save crop_info.json
    crop_info = {}
    patient_to_caseids = defaultdict(list)

    for case_id, patient_id, bbox_info in ok_results:
        crop_info[case_id] = bbox_info
        patient_to_caseids[patient_id].append(case_id)

    crop_info_file = join(ds_root, "crop_info.json")
    with open(crop_info_file, 'w') as f:
        json.dump(crop_info, f, indent=2)

    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Generate dataset.json
    if len(ok_results) > 0:
        num_training_cases = len(ok_results)
        generate_dataset_json(
            output_folder=ds_root,
            channel_names={0: "CT"},
            labels={"background": 0, "lesion": 1},
            num_training_cases=num_training_cases,
            file_ending=".nii.gz",
            dataset_name=dataset_name
        )

    # ==================== Final report ====================
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"{'='*70}")
    print(f"  Succeeded: {len(ok_results)} cases")
    print(f"  Skipped: {len(skip_reasons)} cases")
    print(f"  Patients: {len(patient_to_caseids)}")
    print(f"\nOutput directories:")
    print(f"  Images: {imagestr}")
    print(f"  Labels: {labelstr}")
    print(f"  Crop info: {crop_info_file}")
    print(f"  Dataset config: {join(ds_root, 'dataset.json')}")

    if len(ok_results) > 0:
        print(f"\nNext steps:")
        print(f"  1. Create splits: python create_splits.py -d {ds_root} --folds 5")
        print(f"  2. Preprocess: nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity")

    print(f"{'='*70}\n")

    if skip_reasons:
        print(f"Skipped cases (first 10):")
        for r in skip_reasons[:10]:
            print(f"  - {r}")
        if len(skip_reasons) > 10:
            print(f"  ... and {len(skip_reasons) - 10} more")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrated data preprocessing pipeline: TotalSegmentator segmentation + cropping + nnU-Net format conversion"
    )
    parser.add_argument("-i", "--input", required=True,
                       help="Root directory of raw data")
    parser.add_argument("-o", "--output", required=True,
                       help="Output root directory (nnUNet_raw)")
    parser.add_argument("--dataset_name", required=True,
                       help="Dataset name (e.g. Dataset080_3DPECT)")
    parser.add_argument("--margin", type=int, default=10,
                       help="Bounding box expansion margin (default: 10)")
    parser.add_argument("-n", "--n_processes", type=int, default=4,
                       help="Number of parallel processes (default: 4)")

    args = parser.parse_args()

    print("=" * 70)
    print("Starting script...")
    print("=" * 70)

    try:
        prepare_dataset(
            source_root=args.input,
            output_root=args.output,
            dataset_name=args.dataset_name,
            margin=args.margin,
            n_processes=args.n_processes
        )
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FAILED: script execution error:")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
