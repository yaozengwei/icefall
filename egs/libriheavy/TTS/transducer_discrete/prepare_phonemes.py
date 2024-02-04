#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
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


"""
This file reads the texts in given manifest and save the new cuts with phoneme tokens.
"""

import argparse
import logging
from pathlib import Path

import tacotron_cleaner.cleaners
from lhotse import CutSet, load_manifest
from piper_phonemize import phonemize_espeak, phoneme_ids_espeak
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="small",
        help="""Subset to process. Possible values are 'small', 'medium', 'large',
        'dev', 'test_clean' and 'test_other'""",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/cases_and_punc/manifests_codebooks"),
        help="Path to directory that saves cuts with encoded codebooks.",
    )

    parser.add_argument(
        "--prompt-duration",
        type=float,
        default=3.0,
        help="Duration in seconds of acoustic prompt.",
    )

    return parser


def prepare_phonemes(in_cuts: CutSet):
    new_cuts = []
    for cut in tqdm(in_cuts):
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, len(cut.supervisions)
        text = cut.supervisions[0].text
        # Text normalization
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        # Convert to phonemes
        # 0 = pad
        # 1 = bos
        # 2 = eos
        tokens_list = phonemize_espeak(text, "en-us")
        tokens = []
        for t in tokens_list:
            tokens.extend(t)
        cut.text_tokens = phoneme_ids_espeak(tokens)
        new_cuts.append(cut)

    return CutSet.from_cuts(new_cuts)


def main():
    parser = get_parser()
    args = parser.parse_args()

    subset = args.subset
    assert subset in ["small", "medium", "large", "dev", "test_clean", "test_other"], subset

    manifest_dir = args.manifest_dir
    prompt_duration = args.prompt_duration
    suffix = ".jsonl.gz"
    prefix = f"libriheavy_cuts_{args.subset}_prompt_{prompt_duration}s"

    out_cuts_filename = manifest_dir / f"{prefix}_phone{suffix}"
    if out_cuts_filename.is_file():
        logging.info(f"{out_cuts_filename} already exists - skipping.")
        return

    in_cuts = load_manifest(manifest_dir / f"{prefix}{suffix}")
    out_cuts = prepare_phonemes(in_cuts)

    out_cuts.to_file(out_cuts_filename)
    logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
