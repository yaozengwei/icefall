#!/usr/bin/env python3
# Copyright       2023 Xiaomi Corporation         (Author: Zengwei Yao)
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
import copy
import logging
import random
from pathlib import Path
from tqdm import tqdm

from lhotse import CutSet, load_manifest_lazy
from lhotse.serialization import SequentialJsonlWriter


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
        default=Path("data/upper_no_punc/manifests"),
        help="Path to directory that saves cuts with encoded codebooks.",
    )

    parser.add_argument(
        "--prompt-duration",
        type=float,
        default=10.0,
        help="""Duration in seconds of acoustic prompt.""",
    )

    return parser


def prepare_prompt(
    in_cuts: CutSet,
    cuts_writer: SequentialJsonlWriter,
    prompt_duration: float,
):
    for speaker in tqdm(in_cuts.speakers):
        # filter cuts for each speaker
        cuts = in_cuts.filter(lambda s: s.supervisions[0].speaker == speaker)

        cuts = cuts.to_eager()
        num_cuts = len(cuts)
        if num_cuts < 2:
            logging.info(
                f"Skip for speaker {speaker} since the number of cuts is less than 2"
            )
            continue

        cuts_prompt = cuts.shuffle()

        for cut, cut_p in zip(cuts, cuts_prompt):
            if cut.id == cut_p.id:
                while True:
                    index = random.randint(0, num_cuts - 1)
                    cut_p = cuts[index]
                    if cut.id != cut_p.id:
                        break

            cut = copy.deepcopy(cut)
            cut_p = copy.deepcopy(cut_p)

            cur_prompt_duration = min(prompt_duration, cut_p.duration)
            # select a random start in prompt_cut
            prompt_start = random.uniform(
                0.0, cut_p.duration - cur_prompt_duration
            )
            cut_p.start = cut_p.start + prompt_start
            cut_p.duration = cur_prompt_duration

            cut.prompt_cut = cut_p
            cuts_writer.write(cut, flush=False)


def main():
    parser = get_parser()
    args = parser.parse_args()

    subset = args.subset
    assert subset in ["small", "medium", "large", "dev", "test_clean", "test_other"], subset

    manifest_dir = args.manifest_dir
    suffix = ".jsonl.gz"

    out_cuts_filename = manifest_dir / f"libriheavy_cuts_{args.subset}_with_prompt{suffix}"
    if out_cuts_filename.is_file():
        logging.info(f"{out_cuts_filename} already exists - skipping.")
        return

    logging.info(f"Preparing prompt manifests for subset {subset}")

    in_cuts = load_manifest_lazy(manifest_dir / f"libriheavy_cuts_{subset}{suffix}")
    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)

    prepare_prompt(in_cuts, cuts_writer, args.prompt_duration)

    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
