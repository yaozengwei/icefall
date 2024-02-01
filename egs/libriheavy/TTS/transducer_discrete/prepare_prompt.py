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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from lhotse import CutSet, load_manifest_lazy
from lhotse.manipulation import combine


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
        default=Path("data/upper_no_punc/manifests_codebooks"),
        help="Path to directory that saves cuts with encoded codebooks.",
    )

    parser.add_argument(
        "--prompt-duration",
        type=float,
        default=3.0,
        help="Duration in seconds of acoustic prompt.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="The number of parallel processes.",
    )

    return parser


def add_prompt(cuts: CutSet, prompt_duration: float) -> CutSet:
    """Add prompt_cut for each cut."""
    new_cuts = []
    num_cuts = len(cuts)
    cuts_prompt = cuts.shuffle()
    for cut, cut_p in zip(cuts, cuts_prompt):
        if cut.id == cut_p.id:
            # select another cut_p with different id
            while True:
                index = random.randint(0, num_cuts - 1)
                cut_p = cuts[index]
                if cut.id != cut_p.id:
                    break

        cut = copy.deepcopy(cut)
        cut_p = copy.deepcopy(cut_p)

        cur_prompt_duration = min(prompt_duration, cut_p.duration)
        # select a random start in prompt_cut
        prompt_start = random.uniform(0.0, cut_p.duration - cur_prompt_duration)
        cut_p.id = cut.id + "_prompt"
        cut_p.start = cut_p.start + prompt_start
        cut_p.duration = cur_prompt_duration

        cut.prompt_cut = cut_p
        new_cuts.append(cut)

    return CutSet.from_cuts(new_cuts)


def prepare_prompt(
    in_cuts: CutSet,
    prompt_duration: float,
    executor: Optional[ProcessPoolExecutor] = None,
) -> CutSet:
    """
    Args:
        in_cuts: Input CutSet
        prompt_duration: Duration in seconds as prompt.
        executor: Used to parallelize the feature extraction process.

    Returns:
        A new CutSet with prompt_cut for each cut.
    """
    cuts_per_speaker = {}  # {speaker: List[cut]}
    for cut in tqdm(in_cuts, desc="Grouping cuts by speakers"):
        speaker = cut.supervisions[0].speaker
        if speaker in cuts_per_speaker:
            cuts_per_speaker[speaker].append(cut)
        else:
            cuts_per_speaker[speaker] = [cut]

    for speaker in tqdm(in_cuts.speakers, desc="Filtering cuts groups"):
        if len(cuts_per_speaker[speaker]) < 2:
            cuts_per_speaker.pop(speaker, None)
            logging.info(f"Skip for speaker {speaker} since the number of cuts is less than 2")

    # List[CutSet]
    cuts_per_speaker = [CutSet.from_cuts(cuts) for cuts in cuts_per_speaker.values()]

    out_cuts = []
    with tqdm(total=len(cuts_per_speaker), desc="Adding prompts for each speaker") as pbar:
        if executor is None:
            for cuts in cuts_per_speaker:
                out_cuts.append(add_prompt(cuts, prompt_duration))
                pbar.update(1)
        else:
            futures = [
                executor.submit(add_prompt, cuts, prompt_duration) for cuts in cuts_per_speaker
            ]
            for future in as_completed(futures):
                out_cuts.append(future.result())
                pbar.update(1)

    return combine(out_cuts)


def main():
    parser = get_parser()
    args = parser.parse_args()

    subset = args.subset
    assert subset in ["small", "medium", "large", "dev", "test_clean", "test_other"], subset

    prompt_duration = args.prompt_duration
    manifest_dir = args.manifest_dir
    suffix = ".jsonl.gz"

    out_cuts_filename = manifest_dir / f"libriheavy_cuts_{args.subset}_prompt_{prompt_duration}s{suffix}"
    if out_cuts_filename.is_file():
        logging.info(f"{out_cuts_filename} already exists - skipping.")
        return

    logging.info(f"Preparing prompt manifests for subset {subset}")

    in_cuts = load_manifest_lazy(manifest_dir / f"libriheavy_cuts_{subset}{suffix}")

    executor = None
    if args.num_jobs > 1:
        executor = ProcessPoolExecutor(max_workers=args.num_jobs)

    out_cuts = prepare_prompt(in_cuts, args.prompt_duration, executor)

    out_cuts.to_file(out_cuts_filename)
    logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
