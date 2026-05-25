"""Prepare Seedance packet metadata for TurboT2AV distillation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _join_prompt(video_prompt: str, audio_prompt: str) -> str:
    parts = [video_prompt.strip(), audio_prompt.strip()]
    return " ".join(part for part in parts if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest.jsonl and prompts.txt from a Seedance packet directory."
    )
    parser.add_argument(
        "--packet_root",
        default="/data/datasets/turbodiff_datasets_and_ckpt/seedance_dataset/dance/packet",
        help="Directory containing mapping.csv and dance_dataset/*.mp4.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to --packet_root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_root = Path(args.packet_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else packet_root

    mapping_path = packet_root / "mapping.csv"
    video_dir = packet_root / "dance_dataset"
    manifest_path = output_dir / "manifest.jsonl"
    prompts_path = output_dir / "prompts.txt"

    if not mapping_path.is_file():
        raise FileNotFoundError(f"mapping.csv not found: {mapping_path}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"dance_dataset directory not found: {video_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    missing = 0
    with mapping_path.open("r", encoding="utf-8", newline="") as src, manifest_path.open(
        "w", encoding="utf-8"
    ) as manifest_f, prompts_path.open("w", encoding="utf-8") as prompts_f:
        reader = csv.DictReader(src)
        required_columns = {"video_id", "video_prompt", "audio_prompt"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"mapping.csv is missing columns: {sorted(missing_columns)}")

        for row in reader:
            video_name = row["video_id"].strip()
            video_path = video_dir / video_name
            if not video_path.is_file():
                missing += 1
                continue

            prompt = _join_prompt(row["video_prompt"], row["audio_prompt"])
            if not prompt:
                continue

            payload = {
                "source_index": count,
                "video_name": video_name,
                "prompt": prompt,
                "video_path": str(video_path),
            }
            manifest_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            prompts_f.write(prompt.replace("\n", " ") + "\n")
            count += 1

    print(f"[seedance] manifest: {manifest_path}")
    print(f"[seedance] prompts:  {prompts_path}")
    print(f"[seedance] samples:  {count}")
    if missing:
        print(f"[seedance] missing videos skipped: {missing}")


if __name__ == "__main__":
    main()
