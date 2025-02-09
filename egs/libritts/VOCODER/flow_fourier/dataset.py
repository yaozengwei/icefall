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


from pathlib import Path
from typing import Optional
import numpy as np
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler

torch.set_num_threads(1)


def build_data_loader(
    wav_list_file: Path,
    corpus_dir: Path,
    sampling_rate: int,
    batch_size: int,
    num_workers: int,
    train: bool = False,
    num_samples: Optional[int] = None,
    world_size: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    dataset = LibriTTSDataset(
        wav_list_file=wav_list_file,
        corpus_dir=corpus_dir,
        sampling_rate=sampling_rate,
        train=train,
        num_samples=num_samples,
    )

    shuffle = train

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if sampler is None else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return dataloader


class LibriTTSDataset(Dataset):
    # Based on https://github.com/gemelo-ai/vocos/blob/main/vocos/dataset.py
    def __init__(
        self,
        wav_list_file: str,
        corpus_dir: str,
        sampling_rate: int,
        train: bool = False,
        num_samples: Optional[int] = None,
        apply_effects: bool = True,
    ):
        with open(wav_list_file) as f:
            self.wav_list = f.read().splitlines()
        self.corpus_dir = corpus_dir
        self.sampling_rate = sampling_rate
        if train:
            assert num_samples is not None
        self.train = train
        self.num_samples = num_samples
        self.apply_effects = apply_effects

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_name = self.wav_list[index]

        y, sr = torchaudio.load(os.path.join(self.corpus_dir, file_name))
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)

        if self.apply_effects:
            gain = np.random.uniform(-1, -6) if self.train else -3
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        if self.num_samples is not None:
            if y.size(-1) < self.num_samples:
                pad_length = self.num_samples - y.size(-1)
                padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            elif self.train:
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                y = y[:, start : start + self.num_samples]
            else:
                # During validation, take always the first segment for determinism
                y = y[:, : self.num_samples]

        return y[0], file_name
