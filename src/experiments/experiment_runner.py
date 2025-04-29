"""CLI for launching experiments."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run quantum sparse attention experiments.")
    parser.add_argument("--config", type=Path, help="Path to experiment config JSON")
    args = parser.parse_args()
    print(f"Running with config {args.config}")
    # TODO: load config, train/evaluate model

if __name__ == "__main__":
    main()