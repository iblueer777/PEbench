#!/usr/bin/env python3
"""
Convert the public pulmonary embolism datasets (CAD-PE / FUMPE / READ) to nnU-Net style NIfTI.

Raw layout (--raw):
  CAD-PE/images/001.nrrd            + CAD-PE/rs/0001RefStd.nrrd   (labels 1..6 = clot ids -> binarized)
  FUMPE/CT_scans/PAT001/*.dcm       + FUMPE/GroundTruth/PAT001.mat (key 'Mask', array order x,y,z -> z,y,x)
  READ/images/GE (DICOM files)/01GE + READ/rs/01GE.nii.gz
  READ/images/TOSHIBA (DICOM files)/01TS + READ/rs/01TS.nii.gz

Output (--out): imagesTr/<case>_0000.nii.gz (int16 CT) and labelsTr/<case>.nii.gz (uint8, 0/1)
  CAD-PE 001   -> pe_001_001
  CAD-PE e0032 -> pe_e0032_e0032
  FUMPE PAT001 -> pe_Patient01_Patient01
  READ 01GE    -> 01GE

Orientation handling:
  * CAD-PE: RefStd stores one value per clot (1..6); all non-zero voxels become label 1.
  * FUMPE: the .mat mask is (y, x, z) and its slices follow the DICOM file names (D0001.dcm, ...),
    while GDCM sorts slices by physical position. The mask is reordered into the image slice order,
    which reverses it for the cases whose file name order runs against the position order.
  * READ: the DICOM series (as sorted by GDCM) has z-direction -1 while the rs label NIfTI has +1.
    The label array is index-aligned with the DICOM array, so the image is written with the label's
    origin and direction. Spacing is taken from the DICOM, because the rs NIfTI stores a rounded
    z-spacing (e.g. 0.62598 instead of 0.625) that drifts by ~1 slice over a long series.

Usage:
  python convert_public_pe_to_nifti.py --raw RAW_ROOT --out OUT_DIR
  python convert_public_pe_to_nifti.py --raw ... --out ... --datasets READ --workers 8
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio
import SimpleITK as sitk

RAW_ROOT = Path("raw_data")                 # set by --raw
OUT_DIR = Path("nnunet_converted")          # set by --out


# ----------------------------------------------------------------------------- readers
def read_dicom_series(folder: Path, return_files=False):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise RuntimeError(f"no DICOM series in {folder}")
    # use the series with the most slices if a folder holds several
    files = max((reader.GetGDCMSeriesFileNames(str(folder), s) for s in series_ids), key=len)
    reader.SetFileNames(files)
    img = reader.Execute()
    return (img, list(files)) if return_files else img


def binarize(lbl: sitk.Image, ref: sitk.Image) -> sitk.Image:
    arr = (sitk.GetArrayFromImage(lbl) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


# ----------------------------------------------------------------------------- case lists
def cases_cadpe():
    cases = []
    for img in sorted((RAW_ROOT / "CAD-PE" / "images").glob("*.nrrd")):
        cid = img.stem                                   # 001 / e0032
        lbl_stem = cid.zfill(4) if cid.isdigit() else cid  # 001 -> 0001
        lbl = RAW_ROOT / "CAD-PE" / "rs" / f"{lbl_stem}RefStd.nrrd"
        cases.append(("CAD-PE", f"pe_{cid}_{cid}", img, lbl))
    return cases


def cases_fumpe():
    cases = []
    for d in sorted((RAW_ROOT / "FUMPE" / "CT_scans").iterdir()):
        if not d.is_dir():
            continue
        num = int(re.sub(r"\D", "", d.name))             # PAT001 -> 1
        pid = f"Patient{num:02d}"
        lbl = RAW_ROOT / "FUMPE" / "GroundTruth" / f"{d.name}.mat"
        cases.append(("FUMPE", f"pe_{pid}_{pid}", d, lbl))
    return cases


def cases_read():
    cases = []
    for vendor in ["GE (DICOM files)", "TOSHIBA (DICOM files)"]:
        for d in sorted((RAW_ROOT / "READ" / "images" / vendor).iterdir()):
            if d.is_dir():
                cases.append(("READ", d.name, d, RAW_ROOT / "READ" / "rs" / f"{d.name}.nii.gz"))
    return cases


COLLECTORS = {"CAD-PE": cases_cadpe, "FUMPE": cases_fumpe, "READ": cases_read}


# ----------------------------------------------------------------------------- conversion
def load_case(source, img_path: Path, lbl_path: Path):
    if source == "CAD-PE":
        img = sitk.ReadImage(str(img_path))
        lbl = binarize(sitk.ReadImage(str(lbl_path)), img)

    elif source == "FUMPE":
        img, files = read_dicom_series(img_path, return_files=True)
        mask = sio.loadmat(str(lbl_path))["Mask"]        # (row=y, col=x, slice=z)
        arr = (mask.transpose(2, 0, 1) > 0).astype(np.uint8)
        if arr.shape != sitk.GetArrayFromImage(img).shape:
            raise RuntimeError(f"mask shape {arr.shape} != image {img.GetSize()[::-1]}")
        # mask slice k belongs to the k-th DICOM file by file name, GDCM sorts by position
        by_name = sorted(Path(f).name for f in files)
        arr = np.ascontiguousarray(arr[[by_name.index(Path(f).name) for f in files]])
        lbl = sitk.GetImageFromArray(arr)
        lbl.CopyInformation(img)

    elif source == "READ":
        dcm = read_dicom_series(img_path)
        lbl_raw = sitk.ReadImage(str(lbl_path))
        img_arr = sitk.GetArrayFromImage(dcm)
        if img_arr.shape != sitk.GetArrayFromImage(lbl_raw).shape:
            raise RuntimeError(f"image {dcm.GetSize()} != label {lbl_raw.GetSize()}")
        img = sitk.GetImageFromArray(img_arr)
        img.CopyInformation(lbl_raw)                     # origin/direction from the label header
        img.SetSpacing(dcm.GetSpacing())                 # spacing from DICOM, see module docstring
        lbl = binarize(lbl_raw, img)
    else:
        raise ValueError(source)

    img = sitk.Cast(img, sitk.sitkInt16)
    return img, lbl


def convert_one(case):
    source, name, img_path, lbl_path = case
    out_img = OUT_DIR / "imagesTr" / f"{name}_0000.nii.gz"
    out_lbl = OUT_DIR / "labelsTr" / f"{name}.nii.gz"
    if out_img.exists() and out_lbl.exists():
        return f"[skip] {name}"
    try:
        if not lbl_path.exists():
            return f"[MISSING LABEL] {name}: {lbl_path}"
        img, lbl = load_case(source, img_path, lbl_path)
        sitk.WriteImage(img, str(out_img), useCompression=True)
        sitk.WriteImage(lbl, str(out_lbl), useCompression=True)
        return f"[ok] {source} {name} size={img.GetSize()} fg={int(sitk.GetArrayViewFromImage(lbl).sum())}"
    except Exception as e:
        return f"[ERROR] {name}: {e}"


def _set_paths(raw, out):
    global RAW_ROOT, OUT_DIR
    RAW_ROOT, OUT_DIR = raw, out


def main():
    global RAW_ROOT, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="raw data root (CAD-PE / FUMPE / READ)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (imagesTr / labelsTr)")
    ap.add_argument("--datasets", nargs="+", default=list(COLLECTORS), choices=list(COLLECTORS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RAW_ROOT, OUT_DIR = args.raw, args.out
    (OUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = [c for ds in args.datasets for c in COLLECTORS[ds]()]
    print(f"{len(cases)} cases: " + ", ".join(f"{ds}={sum(c[0] == ds for c in cases)}" for ds in args.datasets))

    # globals don't reach spawned workers on all platforms; pass the paths via initializer
    failed = []
    with Pool(args.workers, initializer=_set_paths, initargs=(RAW_ROOT, OUT_DIR)) as pool:
        for msg in pool.imap_unordered(convert_one, cases):
            print(msg, flush=True)
            if msg.startswith("[ERROR]") or msg.startswith("[MISSING"):
                failed.append(msg)

    print(f"\ndone: {len(cases) - len(failed)}/{len(cases)} cases -> {OUT_DIR}")
    for msg in failed:
        print(" ", msg)


if __name__ == "__main__":
    main()
