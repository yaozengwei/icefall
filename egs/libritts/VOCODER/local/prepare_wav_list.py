#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from pathlib import Path


DATASET_PARTS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default="./LibriTTS",
        help="Root dir to the LibriTTS dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="./data/wav_list",
        help="Dir to save the wav list files.",
    )
    return parser


def prepare_wav_list(root_dir: Path, out_dir: Path):
    for part in DATASET_PARTS:
        logging.info(f"Processing {part}")

        part_dir = root_dir / part
        if not part_dir.is_dir():
            logging.info(f"{part_dir} does not exist - skipping.")
            continue

        cnt = 0
        out_file = out_dir / (part + ".txt")
        with open(out_file, "w") as f:
            for wav_file in part_dir.rglob("*.wav"):
                f.write(str(wav_file.relative_to(part_dir)) + "\n")
                cnt += 1
        logging.info(f"Saved {cnt} lines to {out_file}")

    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_parser().parse_args()
    prepare_wav_list(args.root_dir, args.out_dir)
