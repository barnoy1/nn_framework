from __future__ import annotations

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Supervisely per-image rectangle annotations to COCO JSON with RLE segmentation."
    )
    parser.add_argument(
        "-c",
        "--conf_data",
        type=str,
        required=True,
        help="Path to the configuration data file.",
    )
    parser.add_argument(
        "-r",
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing split folders.",
    )
    parser.add_argument(
        "--experiment_conf",
        type=str,
        default=None,
        help="Optional experiment configuration YAML to copy.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for COCO JSON files.",
    )
    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid"],
        help="Dataset splits.",
    )
    parser.add_argument(
        "--ann_subdir",
        type=str,
        default="ann",
        help="Annotation subdirectory under each split.",
    )
    parser.add_argument(
        "--img_subdir",
        type=str,
        default="img",
        help="Image subdirectory under each split.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="info",
        help="Logging level (debug/info/error).",
    )
    return parser.parse_args()
