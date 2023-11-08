# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2023  Xiaomi Corporation     (Author: Zengwei Yao)
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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import torch
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.dataset.collation import collate_audio
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class SpeechEncodingDataset(torch.utils.data.Dataset):
    """The PyTorch Dataset for the speech encoding task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'cuts': list of Cuts  # when return_cuts=True
        }
    """
    def __init__(self, return_cuts: bool = True, sampling_rate: int = 22050):
        super().__init__()
        self.return_cuts = return_cuts
        self.sampling_rate = sampling_rate

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[Cut]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        # self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        # cuts = cuts.sort_by_duration(ascending=False)

        cuts = cuts.resample(self.sampling_rate)

        audio, audio_lens = collate_audio(cuts)

        batch = {"audio": audio, "audio_lens": audio_lens}

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]
        return batch


class SpeechEncDataModule:
    """DataModule for speech encoding."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Speech encoding data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=600.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--sampling-rate",
            type=int,
            default=22050,
            help="Target sampling rate.",
        )

    def dataloader(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        dataset = SpeechEncodingDataset(
            return_cuts=self.args.return_cuts,
            sampling_rate=self.args.sampling_rate,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            sampler = DynamicBucketingSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=False,
                num_buckets=self.args.num_buckets,
                drop_last=False,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            sampler = SimpleCutSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=False,
                drop_last=False,
            )

        logging.debug("About to create test dataloader")
        dl = DataLoader(
            dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )
        return dl

    @lru_cache()
    def load_subset(self, cuts_filename: Path) -> CutSet:
        return load_manifest_lazy(cuts_filename)
