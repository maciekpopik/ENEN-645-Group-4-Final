from __future__ import annotations

import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

SEED = 42
DRY_RUN = True   # <<< CHANGE TO False WHEN READY TO ACTUALLY COPY
VERBOSE = True

ROOT_TARGET = Path(r"C:\Users\MPopi\OneDrive - University of Calgary\Classes\Term 2 - Winter 2026\ENEN 645\Project\Data\PlantLab2RealGeneralization")
ROOT_PV = Path(r"C:\Users\MPopi\OneDrive - University of Calgary\Classes\Term 2 - Winter 2026\ENEN 645\Project\Data\PlantVillage-Dataset\color")
ROOT_PD = Path(r"C:\Users\MPopi\OneDrive - University of Calgary\Classes\Term 2 - Winter 2026\ENEN 645\Project\Data\PlantDoc-Dataset")
ROOT_OTHER = Path(r"C:\Users\MPopi\OneDrive - University of Calgary\Classes\Term 2 - Winter 2026\ENEN 645\Project\Data\Other")
MAPPING_XLSX = Path(r"C:\Users\MPopi\OneDrive - University of Calgary\Classes\Term 2 - Winter 2026\ENEN 645\Project\Data\FolderNameMap.xlsx")

PV_SPLITS = {
    "Train": 0.70,
    "Val": 0.15,
    "Test_ID": 0.15,
}

OOD_MAIN_SPLIT = 0.90   # 90% Of out-of-distribution set goes toward testing
OOD_FEWSHOT_SPLIT = 0.10  # 10% Of out-of-distribution set goes toward few shot application

# image extensions accepted
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Datasets additional to Plant Doc were also used in the target out-of-distribution set
# This is the min number of samples to consider per class when topping up from other datasets
# Otherwise, the average count from other classes will be used
MIN_TARGET_OOD_PER_CLASS = 5

# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    if VERBOSE:
        print(msg)

def clean_filename(name: str) -> str:
    # remove leading/trailing whitespace
    name = name.strip()

    # remove space before extension: "file .jpg" → "file.jpg"
    import re
    name = re.sub(r"\s+\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", r".\1", name, flags=re.IGNORECASE)

    # optional: collapse multiple spaces
    name = re.sub(r"\s+", " ", name)

    return name

# Helper for file search
def norm_name(s: str) -> str:
    """
    Normalize folder names for tolerant matching.
    """
    s = str(s).strip().lower()
    replacements = {
        " ": "",
        "_": "",
        "-": "",
        ",": "",
        "(": "",
        ")": "",
        ".": "",
        "'": "",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s

# Helper for listing all of the files present within a folder path
def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name.lower())
    return files

# Helper for copying file source to destination (when Dry Run disabled)
def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if DRY_RUN:
        return
    shutil.copy2(src, dst)

# Helper to join file name + prefix
def safe_filename_prefix(prefix: str, fname: str) -> str:
    return f"{prefix}__{fname}"

# Helper Function that shuffes provided paths as per random seed
def deterministic_shuffle(items: List[Path], seed: int) -> List[Path]:
    rng = random.Random(seed)
    items = items.copy()
    rng.shuffle(items)
    return items

# Helper function to split file paths into 3 group sizes as per 'counts'
def split_exact_count(items: List[Path], counts: Tuple[int, int, int]) -> Tuple[List[Path], List[Path], List[Path]]:
    a, b, c = counts
    assert a + b + c == len(items)
    return items[:a], items[a:a+b], items[a+b:a+b+c]

# Returns Plant Village split count as tuple based on PV_SPLITS constant
def compute_pv_split_counts(n: int) -> Tuple[int, int, int]:
    """
    Largest remainder method so counts always sum exactly to n.
    """
    raw = {
        "Train": n * PV_SPLITS["Train"],
        "Val": n * PV_SPLITS["Val"],
        "Test_ID": n * PV_SPLITS["Test_ID"],
    }
    floored = {k: math.floor(v) for k, v in raw.items()}
    remainder = n - sum(floored.values())

    frac_order = sorted(
        raw.keys(),
        key=lambda k: (raw[k] - floored[k]),
        reverse=True
    )

    for i in range(remainder):
        floored[frac_order[i]] += 1

    return floored["Train"], floored["Val"], floored["Test_ID"]

# Similar method to above
# Splits the content of a path list based on OOD_MAIN_SPLIT constant (90%)
# Considers all content in path
def split_90_10(items: List[Path]) -> Tuple[List[Path], List[Path]]:
    n = len(items)
    n_ood = math.floor(n * OOD_MAIN_SPLIT)
    return items[:n_ood], items[n_ood:]


# 90/10 OOD split based on only the files that are to be used
def split_used_subset_90_10(files: List[Path], total_to_use: int) -> Tuple[List[Path], List[Path]]:
    used = files[:max(0, total_to_use)]
    n = len(used)
    n_ood = math.floor(n * OOD_MAIN_SPLIT)
    return used[:n_ood], used[n_ood:]

# Checks if fil exists within specified path wit tolerance on naming
def resolve_folder(root: Path, requested_name: Optional[str]) -> Optional[Path]:
    """
    Try exact match first, then normalized match among immediate child folders.
    """
    if requested_name is None:
        return None

    requested_name = str(requested_name).strip()
    if requested_name == "":
        return None
    if requested_name.lower() == "none found":
        return None

    exact = root / requested_name
    if exact.exists() and exact.is_dir():
        return exact

    wanted = norm_name(requested_name)
    matches = []
    for p in root.iterdir():
        if p.is_dir() and norm_name(p.name) == wanted:
            matches.append(p)

    if len(matches) == 1:
        log(f"[resolve] Fallback matched '{requested_name}' -> '{matches[0].name}' under {root}")
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous normalized match for '{requested_name}' under '{root}': {[m.name for m in matches]}")

    return None

@dataclass
class ClassRow:
    common_name: str
    plant_village: Optional[str]
    plant_doc: Optional[str]
    plant_doc_alt: Optional[str]
    notes: Optional[str]

# ============================================================
# MAIN LOGIC
# ============================================================

# Main logic follows mapping within excel spreadsheet generated manually for like classes
# between Plant Village Dataset and Plant Doc dataset + others
# This program uses that as a map
def load_mapping(xlsx_path: Path) -> List[ClassRow]:
    df = pd.read_excel(xlsx_path)

    # Use column (1/0) defines if this part of map is to be used
    if "Use" in df.columns:
        df = df[df["Use"].fillna(0).astype(int) == 1]

    expected_cols = [
        "Common Name (PV format)",
        "PlantVillage",
        "PlantDoc",
        "PlantDoc Alternative",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column in Excel: {col}")

    # Loop row-by-row and collect al names from excel naming map
    rows: List[ClassRow] = []
    for _, r in df.iterrows():
        common_name = str(r["Common Name (PV format)"]).strip()
        if not common_name or common_name.lower() == "nan":
            continue

        rows.append(
            ClassRow(
                common_name=common_name,
                plant_village=None if pd.isna(r["PlantVillage"]) else str(r["PlantVillage"]).strip(),
                plant_doc=None if pd.isna(r["PlantDoc"]) else str(r["PlantDoc"]).strip(),
                plant_doc_alt=None if pd.isna(r["PlantDoc Alternative"]) else str(r["PlantDoc Alternative"]).strip(),
                notes=None if ("Notes" not in df.columns or pd.isna(r.get("Notes"))) else str(r.get("Notes")).strip(),
            )
        )
    return rows

def validate_paths() -> None:
    required = {
        "ROOT_PV": ROOT_PV,
        "ROOT_PD": ROOT_PD,
        "ROOT_OTHER": ROOT_OTHER,
        "MAPPING_XLSX": MAPPING_XLSX,
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

def ensure_target_layout() -> Dict[str, Path]:
    out = {
        "Train": ROOT_TARGET / "Train",
        "Val": ROOT_TARGET / "Val",
        "Test_ID": ROOT_TARGET / "Test_ID",
        "Test_OOD": ROOT_TARGET / "Test_OOD",
        "Few_Shot": ROOT_TARGET / "Few_Shot",
        "Logs": ROOT_TARGET / "_logs",
    }
    for p in out.values():
        if not DRY_RUN:
            p.mkdir(parents=True, exist_ok=True)
    return out

def collect_pd_primary(row: ClassRow) -> Tuple[List[Path], List[Path], Dict[str, int]]:
    """
    Combine PlantDoc train + test for the mapped class, then split 90/10.
    """
    stats = {
        "pd_train_total": 0,
        "pd_test_total": 0,
        "pd_primary_total": 0,
    }

    if not row.plant_doc or row.plant_doc.lower() == "none found":
        return [], [], stats

    train_folder = resolve_folder(ROOT_PD / "train", row.plant_doc)
    test_folder = resolve_folder(ROOT_PD / "test", row.plant_doc)

    train_files = list_image_files(train_folder) if train_folder else []
    test_files = list_image_files(test_folder) if test_folder else []

    stats["pd_train_total"] = len(train_files)
    stats["pd_test_total"] = len(test_files)
    stats["pd_primary_total"] = len(train_files) + len(test_files)

    all_files = train_files + test_files
    all_files = deterministic_shuffle(all_files, seed=SEED + abs(hash(row.common_name)) % 10_000_000)

    return split_90_10(all_files)[0], split_90_10(all_files)[1], stats

def collect_alt(row: ClassRow) -> Tuple[List[Path], Dict[str, int]]:
    stats = {
        "alt_total": 0,
    }

    if not row.plant_doc_alt or row.plant_doc_alt.lower() == "none found":
        return [], stats

    alt_folder = resolve_folder(ROOT_OTHER, row.plant_doc_alt)
    alt_files = list_image_files(alt_folder) if alt_folder else []
    stats["alt_total"] = len(alt_files)

    alt_files = deterministic_shuffle(
        alt_files,
        seed=SEED + 1_000_000 + abs(hash(row.common_name)) % 10_000_000
    )

    return alt_files, stats

def copy_group(files: List[Path], dest_dir: Path, source_tag: str, manifest_rows: List[dict], class_name: str, split_name: str) -> int:
    if not files:
        return 0

    copied = 0
    for src in files:
        clean_name = clean_filename(src.name)
        if src.name != clean_name:
            log(f"[rename] '{src.name}' -> '{clean_name}'")

        dst_name = safe_filename_prefix(source_tag, clean_name)
        dst = dest_dir / class_name / dst_name
        copy_file(src, dst)
        copied += 1

        manifest_rows.append({
            "split": split_name,
            "class_name": class_name,
            "source_tag": source_tag,
            "src_path": str(src),
            "dst_path": str(dst),
        })

    return copied

def main() -> None:
    validate_paths()
    targets = ensure_target_layout()
    rows = load_mapping(MAPPING_XLSX)

    log(f"Loaded {len(rows)} mapping rows from Excel.")
    if DRY_RUN:
        log("DRY_RUN = True -> no files will be copied.")
    else:
        log("DRY_RUN = False -> files WILL be copied.")

    manifest_rows: List[dict] = []
    summary_rows: List[dict] = []
    warnings: List[str] = []

    # --------------------------------------------------------
    # Pass 1: inspect PlantDoc-supported classes to compute a
    # target OOD count for top-up from Other
    # --------------------------------------------------------
    primary_ood_counts = []

    primary_cache: Dict[str, Tuple[List[Path], List[Path], Dict[str, int]]] = {}
    alt_cache: Dict[str, Tuple[List[Path], List[Path], Dict[str, int]]] = {}

    for row in rows:
        pd_ood, pd_few, pd_stats = collect_pd_primary(row)
        alt_files, alt_stats = collect_alt(row)

        primary_cache[row.common_name] = (pd_ood, pd_few, pd_stats)
        alt_cache[row.common_name] = (alt_files, alt_stats)

        if len(pd_ood) > 0:
            primary_ood_counts.append(len(pd_ood))

    if primary_ood_counts:
        target_ood_count = max(MIN_TARGET_OOD_PER_CLASS, round(sum(primary_ood_counts) / len(primary_ood_counts)))
    else:
        target_ood_count = MIN_TARGET_OOD_PER_CLASS

    log(f"Computed target OOD class size for top-up: {target_ood_count}")

    # --------------------------------------------------------
    # Pass 2: process each class
    # --------------------------------------------------------
    for idx, row in enumerate(rows, start=1):
        common = row.common_name
        log(f"\n[{idx}/{len(rows)}] Processing: {common}")

        # ---------------------------
        # PV -> Train / Val / Test_ID
        # ---------------------------
        pv_folder = resolve_folder(ROOT_PV, row.plant_village)
        if pv_folder is None:
            warnings.append(f"[PV MISSING] Could not resolve PlantVillage folder for class '{common}' from '{row.plant_village}'")
            pv_files = []
        else:
            pv_files = list_image_files(pv_folder)

        pv_files = deterministic_shuffle(pv_files, seed=SEED + 2_000_000 + abs(hash(common)) % 10_000_000)
        n_pv = len(pv_files)
        n_train, n_val, n_test_id = compute_pv_split_counts(n_pv)

        pv_train, pv_val, pv_test_id = split_exact_count(pv_files, (n_train, n_val, n_test_id))

        copied_train = copy_group(pv_train, targets["Train"], "PV", manifest_rows, common, "Train")
        copied_val = copy_group(pv_val, targets["Val"], "PV", manifest_rows, common, "Val")
        copied_test_id = copy_group(pv_test_id, targets["Test_ID"], "PV", manifest_rows, common, "Test_ID")

        # ---------------------------
        # PlantDoc primary + Other alt
        # ---------------------------
        pd_ood, pd_few, pd_stats = primary_cache[common]
        alt_files, alt_stats = alt_cache[common]

        # Use all available PlantDoc 90% into Test_OOD.
        selected_pd_ood = pd_ood
        selected_pd_few = pd_few

        # Top-up logic:
        # 1) If class has PlantDoc data, keep all PlantDoc OOD data.
        # 2) If total PlantDoc OOD count is below target_ood_count and alt exists,
        #    add enough alt OOD files to reach target.
        # 3) If class has no PlantDoc but has alt only, sample alt OOD up to target_ood_count.
        if len(selected_pd_ood) > 0:
            need = max(0, target_ood_count - len(selected_pd_ood))
            alt_total_to_use = math.ceil(need / OOD_MAIN_SPLIT) if need > 0 else 0
        else:
            alt_total_to_use = math.ceil(target_ood_count / OOD_MAIN_SPLIT) if len(alt_files) > 0 else 0

        # Split ONLY the used subset into 90/10
        selected_alt_ood, selected_alt_few = split_used_subset_90_10(alt_files, alt_total_to_use)

        copied_test_ood_pd = 0
        copied_test_ood_alt = 0
        copied_few_pd = 0
        copied_few_alt = 0

        if len(selected_pd_ood) > 0:
            copied_test_ood_pd = copy_group(selected_pd_ood, targets["Test_OOD"], "PD", manifest_rows, common, "Test_OOD")
            copied_few_pd = copy_group(selected_pd_few, targets["Few_Shot"], "PD", manifest_rows, common, "Few_Shot")

        if len(selected_alt_ood) > 0:
            copied_test_ood_alt = copy_group(selected_alt_ood, targets["Test_OOD"], "ALT", manifest_rows, common, "Test_OOD")
            copied_few_alt = copy_group(selected_alt_few, targets["Few_Shot"], "ALT", manifest_rows, common, "Few_Shot")

        summary_rows.append({
            "class_name": common,
            "pv_folder_requested": row.plant_village,
            "pd_folder_requested": row.plant_doc,
            "alt_folder_requested": row.plant_doc_alt,
            "pv_total": n_pv,
            "pv_train": copied_train,
            "pv_val": copied_val,
            "pv_test_id": copied_test_id,
            "pd_train_total": pd_stats["pd_train_total"],
            "pd_test_total": pd_stats["pd_test_total"],
            "pd_primary_total": pd_stats["pd_primary_total"],
            "pd_ood_used": copied_test_ood_pd,
            "pd_few_used": copied_few_pd,
            "alt_total": alt_stats["alt_total"],
            "alt_ood_used": copied_test_ood_alt,
            "alt_few_used": copied_few_alt,
            "test_ood_total_used": copied_test_ood_pd + copied_test_ood_alt,
            "few_shot_total_used": copied_few_pd + copied_few_alt,
            "notes": row.notes,
        })

        if n_pv == 0:
            warnings.append(f"[EMPTY PV] {common} has 0 PlantVillage images.")
        if (copied_test_ood_pd + copied_test_ood_alt) == 0:
            warnings.append(f"[EMPTY OOD] {common} has 0 Test_OOD images.")
        if row.plant_doc and copied_test_ood_pd == 0:
            warnings.append(f"[PD UNRESOLVED OR EMPTY] '{common}' requested PlantDoc folder '{row.plant_doc}' but yielded 0 OOD files.")
        if row.plant_doc_alt and row.plant_doc_alt.lower() != "none found" and copied_test_ood_alt == 0 and copied_few_alt == 0:
            warnings.append(f"[ALT UNRESOLVED OR EMPTY] '{common}' requested alt folder '{row.plant_doc_alt}' but yielded 0 files.")

    # --------------------------------------------------------
    # Write logs
    # --------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    warnings_df = pd.DataFrame({"warning": warnings})

    log("\n=== SUMMARY ===")
    log(summary_df[[
        "class_name",
        "pv_total", "pv_train", "pv_val", "pv_test_id",
        "pd_primary_total", "pd_ood_used", "pd_few_used",
        "alt_total", "alt_ood_used", "alt_few_used",
        "test_ood_total_used", "few_shot_total_used"
    ]].to_string(index=False))

    if not DRY_RUN:
        targets["Logs"].mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(targets["Logs"] / "dataset_build_summary.csv", index=False)
        manifest_df.to_csv(targets["Logs"] / "dataset_build_manifest.csv", index=False)
        warnings_df.to_csv(targets["Logs"] / "dataset_build_warnings.csv", index=False)

    print("\nDone.")
    if warnings:
        print(f"There were {len(warnings)} warning(s).")
        for w in warnings:
            print(" -", w)

if __name__ == "__main__":
    try:
        main()

        # --------------------------------------------------------
        # ZIP OUTPUT DATASET
        # --------------------------------------------------------
        if not DRY_RUN:
            import zipfile

            zip_path = ROOT_TARGET / "PlantLab2RealGeneralization.zip"

            print(f"\nCreating zip archive at: {zip_path}")

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in ROOT_TARGET.rglob("*"):

                    # Skip the zip itself if re-running
                    if path == zip_path:
                        continue

                    # Skip logs folder except summary file
                    if "_logs" in path.parts:
                        if path.name == "dataset_build_summary.csv":
                            arcname = Path("dataset_build_summary.csv")
                            zf.write(path, arcname)
                        continue

                    if path.is_file():
                        # preserve relative structure inside zip
                        arcname = path.relative_to(ROOT_TARGET)
                        zf.write(path, arcname)

            print("Zip archive created successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)