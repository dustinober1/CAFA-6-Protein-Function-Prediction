#!/usr/bin/env python3
"""
Main entry point for the CAFA-6 protein function prediction pipeline.

This script runs the complete training and prediction pipeline for the
CAFA-6 protein function prediction competition.

Usage:
    python run_pipeline.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_pipeline import ComprehensiveTrainingPipeline
import argparse


def main():
    """Run the complete training and prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="CAFA-6 Protein Function Prediction Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default="cafa-6-protein-function-prediction",
        help="Path to data directory (default: cafa-6-protein-function-prediction)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Path to output directory (default: results)",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = ComprehensiveTrainingPipeline(args.data_dir, args.output_dir)

    # Run full pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
