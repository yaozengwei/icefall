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

import logging
import os
import argparse

from kaldi_native_io import WaveWriter, read_wave


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--splits", default="train,val", help="which splits to write")

    parser.add_argument(
        "--wav-list-file",
        type=str,
        default="data/wav_list/train-full-960.txt",
        help="Wav list file of dataset.",
    )

    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="download/libritts/LibriTTS",
        help="Root dir to the LibriTTS dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/scp_ark",
        help="Dir to save the scp/ark files.",
    )

    parser.add_argument(
        "--num-per-split",
        type=int,
        default=10000,
        help="Number of samples per split.",
    )

    return parser.parse_args()


def write_dataset(
    wav_list_file: str,
    corpus_dir: str,
    output_dir: str,
    num_per_split: int,
):

    with open(wav_list_file) as f:
        wav_list = f.read().splitlines()

    num_wavs = len(wav_list)
    logging.info(f"Total number of samples: {num_wavs}")

    all_keys = set()
    num_finised = 0
    num_splits = (num_wavs + num_per_split - 1) // num_per_split

    for split_idx in range(num_splits):
        base = os.path.join(output_dir, "{:06d}".format(split_idx))
        wspecifier = f"ark,scp:{base}.ark,{base}.scp"
        with WaveWriter(wspecifier) as ko:
            for local_i in range(num_per_split):
                global_i = split_idx * num_per_split + local_i
                if global_i >= num_wavs:
                    break  # reach the end

                file_name = wav_list[global_i]
                wav_file = os.path.join(corpus_dir, file_name)
                audio = read_wave(wav_file)

                # Construct a uniqu key using the filename.
                key = file_name

                # Useful check.
                assert key not in all_keys
                all_keys.add(key)

                ko.write(key, audio)

                num_finised += 1

        logging.info(f"Writen to {wspecifier}")
        logging.info(f"Finished {num_finised} samples so far..")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)
    write_dataset(args.wav_list_file, args.corpus_dir, args.output_dir, args.num_per_split)
