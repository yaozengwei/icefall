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
        "--prompt-num-codebook-frames",
        type=int,
        default=225,
        help="""Number of codebook frames for acoustic prompt. E.g, with
        sampling_rate=24000, codebook_frame_shift=320, the prompt audio duration for
        prompt_num_codebooks_frames=225 is 225 * 320 / 24000 = 3.0 seconds.""",
    )

    return parser


def prepare_prompt(
    in_cuts: CutSet,
    cuts_writer: SequentialJsonlWriter,
    prompt_num_codebook_frames: int,
):

    for speaker in in_cuts.speakers:
        # filter cuts for one speaker
        cuts = in_cuts.filter(lambda s: s.supervisions[0].speaker == speaker)
        # shuffle the cuts to generate prompt pairs, using cuts[i+1] as prompt of cuts[i]
        cuts = cuts.to_eager().shuffle()

        num_cuts = len(cuts)
        for i in range(num_cuts):
            cut = copy.deepcopy(cuts[i])
            # use next cut as prompt cut
            prompt_cut = cuts[(i + 1) % num_cuts]
            cut.prompt_cut = prompt_cut
            assert prompt_cut.codebooks.num_frames >= prompt_num_codebook_frames, (
                prompt_cut.id,
                prompt_cut.codebooks.num_frames,
                prompt_num_codebook_frames
            )
            # select a random frame start
            cut.prompt_codebook_frame_start = random.randint(
                0, prompt_cut.codebooks.num_frames - prompt_num_codebook_frames
            )
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

    logging.info("Preparing prompt manifests for subset")

    in_cuts = load_manifest_lazy(manifest_dir / f"libriheavy_cuts_{subset}{suffix}")
    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)

    in_cuts = in_cuts.filter(
        lambda s: s.codebooks.num_frames >= args.prompt_num_codebook_frames
    )
    prepare_prompt(in_cuts, cuts_writer, args.prompt_num_codebook_frames)

    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
