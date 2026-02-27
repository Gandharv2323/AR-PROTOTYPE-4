# Module: validate_dataset
# License: MIT (ARVTON project)
# Description: Dataset validation — checks manifest completeness, file integrity, license compliance.
# Platform: Both (Colab + AMD ROCm + Local)
# Dependencies: json, os, pathlib, Pillow, numpy, colorama

"""
Dataset Validation
===================
Validates the final train/val manifests and prints a comprehensive
readiness report including pair counts, source distribution, category
distribution, missing files, and license compliance.

Usage:
    python -m pipeline.validate_dataset --config configs/dataset_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.platform_utils import (
    detect_platform,
    get_paths,
    load_config,
    setup_logging,
)

logger = logging.getLogger("arvton.validate_dataset")

# ═══════════════════════════════════════════════════════════════════════
# COMMERCIAL LICENSE REGISTRY
# ═══════════════════════════════════════════════════════════════════════

COMMERCIAL_LICENSES = {
    "MIT",
    "Apache-2.0",
    "Apache 2.0",
    "CC0",
    "CC0 1.0",
    "CC BY 4.0",
    "CC-BY-4.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "Public Domain",
}

NON_COMMERCIAL_LICENSES = {
    "CC BY-NC 4.0",
    "CC BY-NC-SA 4.0",
    "CC BY-NC-ND 4.0",
    "CC BY-NC",
    "CC BY-NC-SA",
    "CC BY-NC-ND",
    "AGPL",
    "GPL",
    "VITON-HD",
    "DressCode",
}

FORBIDDEN_SOURCES = {
    "VITON-HD",
    "viton-hd",
    "DressCode",
    "dresscode",
}


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single dataset record for completeness and file existence.

    Args:
        record: A manifest record dict.

    Returns:
        Validation result dict with keys:
            - valid (bool)
            - warnings (list of str)
            - errors (list of str)
            - record_id (str)
    """
    result = {
        "record_id": record.get("id", "UNKNOWN"),
        "valid": True,
        "warnings": [],
        "errors": [],
    }

    # Check required fields
    required_fields = [
        "id", "person_image", "person_mask", "body_parse",
        "densepose", "garment_image", "garment_mask", "category",
        "license", "source",
    ]

    for field in required_fields:
        if field not in record:
            result["errors"].append(f"Missing required field: {field}")
            result["valid"] = False
        elif not record[field] and field not in ["license", "source"]:
            result["warnings"].append(f"Empty field: {field}")

    # Check file existence for path fields
    file_fields = [
        "person_image", "person_mask", "body_parse",
        "densepose", "garment_image", "garment_mask",
    ]
    for field in file_fields:
        path = record.get(field, "")
        if path and not Path(path).exists():
            result["errors"].append(f"File not found: {field} = {path}")
            result["valid"] = False

    # Check license compliance
    record_license = record.get("license", "")
    if record_license in NON_COMMERCIAL_LICENSES:
        result["errors"].append(
            f"NON-COMMERCIAL license detected: {record_license}"
        )
        result["valid"] = False

    # Check source compliance
    source = record.get("source", "")
    if source in FORBIDDEN_SOURCES:
        result["errors"].append(
            f"FORBIDDEN source detected: {source} — "
            "VITON-HD and DressCode are NOT commercially licensed"
        )
        result["valid"] = False

    # Check category
    category = record.get("category", "")
    if category not in ["upper", "lower", "dress"]:
        result["warnings"].append(f"Non-standard category: {category}")

    return result


def validate_manifest(
    manifest_path: str,
    check_images: bool = True,
) -> Dict[str, Any]:
    """
    Validate an entire manifest file.

    Args:
        manifest_path: Path to the JSON manifest file.
        check_images: If True, also verify image file existence and readability.

    Returns:
        Comprehensive validation report dict.
    """
    manifest_path = Path(manifest_path)

    report = {
        "manifest_path": str(manifest_path),
        "exists": manifest_path.exists(),
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "by_source": Counter(),
        "by_category": Counter(),
        "by_license": Counter(),
        "missing_files": [],
        "license_violations": [],
        "source_violations": [],
        "warnings": [],
        "errors": [],
        "license_compliance": "UNKNOWN",
        "ready_for_training": False,
    }

    if not manifest_path.exists():
        report["errors"].append(f"Manifest file not found: {manifest_path}")
        report["license_compliance"] = "FAIL"
        return report

    # Load manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        report["errors"].append(f"Invalid JSON: {e}")
        return report

    report["total_records"] = len(records)

    # Validate each record
    for record in records:
        vr = validate_record(record)

        if vr["valid"]:
            report["valid_records"] += 1
        else:
            report["invalid_records"] += 1

        report["warnings"].extend(vr["warnings"])
        report["errors"].extend(vr["errors"])

        # Collect stats
        report["by_source"][record.get("source", "unknown")] += 1
        report["by_category"][record.get("category", "unknown")] += 1
        report["by_license"][record.get("license", "unknown")] += 1

        # Track specific violations
        source = record.get("source", "")
        if source in FORBIDDEN_SOURCES:
            report["source_violations"].append(record.get("id", "?"))

        lic = record.get("license", "")
        if lic in NON_COMMERCIAL_LICENSES:
            report["license_violations"].append(
                f"{record.get('id', '?')} ({lic})"
            )

        # Track missing files
        for field in ["person_image", "person_mask", "body_parse",
                      "densepose", "garment_image", "garment_mask"]:
            path = record.get(field, "")
            if path and not Path(path).exists():
                report["missing_files"].append(path)

    # Determine compliance
    if report["license_violations"] or report["source_violations"]:
        report["license_compliance"] = "FAIL"
    else:
        report["license_compliance"] = "PASS"

    # Determine training readiness
    report["ready_for_training"] = (
        report["valid_records"] >= 100  # Minimum viable
        and report["license_compliance"] == "PASS"
        and not report["source_violations"]
    )

    return report


def validate_image_quality(
    manifest_path: str,
    sample_size: int = 50,
) -> Dict[str, Any]:
    """
    Spot-check image quality on a random sample of records.

    Args:
        manifest_path: Path to the manifest.
        sample_size: Number of records to sample.

    Returns:
        Image quality report dict.
    """
    import numpy as np
    from PIL import Image

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    np.random.seed(42)
    sample_indices = np.random.choice(
        len(records), size=min(sample_size, len(records)), replace=False
    )
    sample = [records[i] for i in sample_indices]

    quality_report = {
        "samples_checked": len(sample),
        "readable": 0,
        "unreadable": 0,
        "min_resolution": None,
        "max_resolution": None,
        "avg_resolution": None,
        "corrupt_files": [],
    }

    resolutions = []
    for record in sample:
        person_path = record.get("person_image", "")
        if not person_path or not Path(person_path).exists():
            continue

        try:
            img = Image.open(person_path)
            img.verify()  # Check for corruption
            img = Image.open(person_path)  # Re-open after verify
            w, h = img.size
            resolutions.append((w, h))
            quality_report["readable"] += 1
        except Exception:
            quality_report["unreadable"] += 1
            quality_report["corrupt_files"].append(person_path)

    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        quality_report["min_resolution"] = f"{min(widths)}x{min(heights)}"
        quality_report["max_resolution"] = f"{max(widths)}x{max(heights)}"
        quality_report["avg_resolution"] = f"{int(np.mean(widths))}x{int(np.mean(heights))}"

    return quality_report


# ═══════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ═══════════════════════════════════════════════════════════════════════


def print_validation_report(
    train_report: Dict[str, Any],
    val_report: Optional[Dict[str, Any]] = None,
    quality_report: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Print a formatted validation report.

    Args:
        train_report: Report from validate_manifest() for training set.
        val_report: Report from validate_manifest() for validation set.
        quality_report: Report from validate_image_quality().

    Returns:
        bool: True if dataset is ready for training.
    """
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        GREEN = Fore.GREEN
        RED = Fore.RED
        YELLOW = Fore.YELLOW
        CYAN = Fore.CYAN
        BOLD = Style.BRIGHT
        RESET = Style.RESET_ALL
    except ImportError:
        GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

    total_records = train_report["total_records"]
    if val_report:
        total_records += val_report["total_records"]

    print()
    print(f"{BOLD}{'=' * 65}")
    print(f"  ARVTON Dataset Validation Report")
    print(f"{'=' * 65}{RESET}")
    print()

    # Total pairs
    print(f"  {'Total pairs:':<25} {CYAN}{total_records}{RESET}")
    print(f"    {'Train:':<23} {train_report['total_records']}")
    if val_report:
        print(f"    {'Validation:':<23} {val_report['total_records']}")
    print()

    # By source
    all_sources = Counter(train_report["by_source"])
    if val_report:
        all_sources += Counter(val_report["by_source"])

    print(f"  {'By source:':<25}")
    for source, count in sorted(all_sources.items()):
        print(f"    {source + ':':<23} {count}")
    print()

    # By category
    all_categories = Counter(train_report["by_category"])
    if val_report:
        all_categories += Counter(val_report["by_category"])

    print(f"  {'By category:':<25}")
    for cat, count in sorted(all_categories.items()):
        print(f"    {cat + ':':<23} {count}")
    print()

    # Missing files
    all_missing = train_report.get("missing_files", [])
    if val_report:
        all_missing += val_report.get("missing_files", [])
    missing_count = len(all_missing)

    if missing_count == 0:
        print(f"  {'Missing files:':<25} {GREEN}None ✓{RESET}")
    else:
        print(f"  {'Missing files:':<25} {RED}{missing_count} files{RESET}")
        for path in all_missing[:10]:
            print(f"    {RED}✗ {path}{RESET}")
        if missing_count > 10:
            print(f"    ... and {missing_count - 10} more")
    print()

    # License compliance
    compliance = train_report["license_compliance"]
    if val_report and val_report["license_compliance"] == "FAIL":
        compliance = "FAIL"

    if compliance == "PASS":
        print(f"  {'License compliance:':<25} {GREEN}PASS ✓{RESET}")
    else:
        print(f"  {'License compliance:':<25} {RED}FAIL ✗{RESET}")
        for violation in train_report.get("license_violations", []):
            print(f"    {RED}✗ {violation}{RESET}")
        for violation in train_report.get("source_violations", []):
            print(f"    {RED}✗ Forbidden source: {violation}{RESET}")
    print()

    # Image quality
    if quality_report:
        print(f"  {'Image quality (sample):':<25}")
        print(f"    {'Samples checked:':<23} {quality_report['samples_checked']}")
        print(f"    {'Readable:':<23} {quality_report['readable']}")
        print(f"    {'Unreadable:':<23} {quality_report['unreadable']}")
        if quality_report["avg_resolution"]:
            print(f"    {'Min resolution:':<23} {quality_report['min_resolution']}")
            print(f"    {'Max resolution:':<23} {quality_report['max_resolution']}")
            print(f"    {'Avg resolution:':<23} {quality_report['avg_resolution']}")
        print()

    # Training readiness
    ready = train_report["ready_for_training"]
    if val_report:
        ready = ready and val_report.get("ready_for_training", True)

    # Override: check minimum 3000 pairs target
    target_met = total_records >= 3000

    print(f"  {'Target (3000+ pairs):':<25}", end="")
    if target_met:
        print(f"{GREEN}MET ({total_records} pairs) ✓{RESET}")
    else:
        print(f"{YELLOW}NOT MET ({total_records}/3000 pairs){RESET}")
    print()

    if ready:
        print(f"  {BOLD}{GREEN}╔═══════════════════════════════════╗")
        print(f"  ║  Ready for training:  YES  ✓      ║")
        print(f"  ╚═══════════════════════════════════╝{RESET}")
    else:
        print(f"  {BOLD}{RED}╔═══════════════════════════════════╗")
        print(f"  ║  Ready for training:  NO   ✗      ║")
        print(f"  ╚═══════════════════════════════════╝{RESET}")

    print()
    print(f"{'=' * 65}")

    return ready


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════


def main():
    """
    Main entry point for dataset validation.

    Usage:
        python -m pipeline.validate_dataset [--config PATH] [--check-images]
    """
    parser = argparse.ArgumentParser(
        description="ARVTON — Dataset Validation",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to dataset_config.yaml")
    parser.add_argument("--platform", type=str, default=None, choices=["colab", "amd", "local"])
    parser.add_argument("--check-images", action="store_true",
                        help="Run image quality spot-checks (slower)")
    parser.add_argument("--train-manifest", type=str, default=None,
                        help="Direct path to train_manifest.json")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="Direct path to val_manifest.json")
    args = parser.parse_args()

    # Setup
    setup_logging("INFO")

    # Resolve paths
    if args.train_manifest:
        train_path = args.train_manifest
        val_path = args.val_manifest
    else:
        config = load_config(args.config)
        platform = args.platform or detect_platform()
        paths = get_paths(platform)
        train_path = str(paths["train_manifest"])
        val_path = str(paths["val_manifest"])

    # Validate training manifest
    logger.info("Validating training manifest: %s", train_path)
    train_report = validate_manifest(train_path)

    # Validate validation manifest (if exists)
    val_report = None
    if val_path and Path(val_path).exists():
        logger.info("Validating validation manifest: %s", val_path)
        val_report = validate_manifest(val_path)

    # Optional image quality check
    quality_report = None
    if args.check_images and Path(train_path).exists():
        logger.info("Running image quality spot-checks...")
        quality_report = validate_image_quality(train_path)

    # Print report
    ready = print_validation_report(train_report, val_report, quality_report)

    # Exit code
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
