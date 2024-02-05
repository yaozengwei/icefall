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
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import MonoCut
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.dataset.collation import collate_audio, collate_custom_field
from lhotse.utils import fix_random_seed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the discrete speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'text': List[str] of len B  # when return_text=True
            'audio': (B x NumSamples) float tensor  # when return_audio=True
            'audio_lens': (B, ) int tensor
            'prompt_audio': (B x NumSamples) float tensor  # when return_prompt_audio=True
            'prompt_audio_lens': (B, ) int tensor
            'text_tokens': (B, NumTokens), int tensor  # when return_text_tokens=True
            'text_tokens_lens': (B, ), int tensor
            'audio_tokens': (B, NumTokens), int tensor  # when return_audio_tokens=True
            'audio_tokens_lens': (B, ), int tensor
            'prompt_audio_tokens': (B, NumTokens), int tensor  # when return_prompt_audio_tokens=True
            'prompt_audio_tokens_lens': (B, ), int tensor
            'speakers': List[str] of len B  # when return_spk_ids=True
            'cut': List of Cuts  # when return_cuts=True
        }
    """

    def __init__(
        self,
        return_text: bool = True,
        return_audio: bool = True,
        return_prompt_audio: bool = True,
        return_text_tokens: bool = False,
        return_audio_tokens: bool = False,
        return_prompt_audio_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
        text_blank_id: int = 0,
        audio_blank_id: int = 0,
    ) -> None:
        super().__init__()

        self.audio_blank_id = audio_blank_id
        self.text_blank_id = text_blank_id

        self.return_text = return_text
        self.return_audio = return_audio
        self.return_prompt_audio = return_prompt_audio

        self.return_text_tokens = return_text_tokens
        self.return_audio_tokens = return_audio_tokens
        self.return_prompt_audio_tokens = return_prompt_audio_tokens

        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        batch = {}

        if self.return_text:
            text = [cut.supervisions[0].text for cut in cuts]
            batch["text"] = text

        if self.return_audio:
            audio, audio_lens = collate_audio(cuts)
            batch["audio"] = audio
            batch["audio_lens"] = audio_lens

        if self.return_prompt_audio:
            prompt_cuts = CutSet.from_cuts(
                [MonoCut.from_dict(cut.prompt_cut) for cut in cuts]
            )
            prompt_audio, prompt_audio_lens = collate_audio(prompt_cuts)
            batch["prompt_audio"] = prompt_audio
            batch["prompt_audio_lens"] = prompt_audio_lens

        if self.return_text_tokens:
            text_tokens = pad_sequence(
                [torch.tensor(cut.text_tokens, dtype=torch.int32) for cut in cuts],
                batch_first=True,
                padding_value=self.text_blank_id,
            )
            text_tokens_lens = torch.tensor(
                [len(cut.text_tokens) for cut in cuts], dtype=torch.int32
            )
            batch["text_tokens"] = text_tokens
            batch["text_tokens_lens"] = text_tokens_lens

        if self.return_audio_tokens:
            audio_tokens, audio_tokens_lens = collate_custom_field(
                cuts, field="codebooks", pad_value=self.audio_blank_id
            )
            batch["audio_tokens"] = audio_tokens
            batch["audio_tokens_lens"] = audio_tokens_lens

        if self.return_prompt_audio_tokens:
            prompt_cuts = CutSet.from_cuts(
                [MonoCut.from_dict(cut.prompt_cut) for cut in cuts]
            )
            prompt_audio_tokens, prompt_audio_tokens_lens = collate_custom_field(
                prompt_cuts, field="codebooks", pad_value=self.audio_blank_id
            )
            batch["prompt_audio_tokens"] = prompt_audio_tokens
            batch["prompt_audio_tokens_lens"] = prompt_audio_tokens_lens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        return batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriHeavyTtsDataModule:
    """DataModule for speech synthesis."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Speech synthesis data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--subset",
            type=str,
            default="S",
            help="""The subset to be used. Should be S, M or L. Note: S subset
            includes libriheavy_cuts_small.jsonl.gz, M subset includes
            libriheavy_cuts_small.jsonl.gz and libriheavy_cuts_medium.jsonl.gz,
            L subset includes libriheavy_cuts_small.jsonl.gz,
            libriheavy_cuts_medium.jsonl.gz and libriheavy_cuts_large.jsonl.gz.
            """,
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests_codebooks"),
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
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
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
            "--return-text",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['text'].",
        )
        group.add_argument(
            "--return-audio",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "fields: batch['audio'] and batch['audio_lens'].",
        )
        group.add_argument(
            "--return-prompt-audio",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "fields: batch['prompt_audio'] and batch['prompt_audio_lens'].",
        )
        group.add_argument(
            "--return-text-tokens",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['text_tokens'].",
        )
        group.add_argument(
            "--return-audio-tokens",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "fields: batch['audio_tokens'] and batch['audio_tokens_lens'].",
        )
        group.add_argument(
            "--return-prompt-audio-tokens",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['prompt_audio_tokens'] and batch['prompt_audio_tokens_lens'].",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--prompt-duration",
            type=float,
            default=3.0,
            help="Duration in seconds of acoustic prompt.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        dataset = SpeechSynthesisDataset(
            return_text=self.args.return_text,
            return_audio=self.args.return_audio,
            return_prompt_audio=self.args.return_prompt_audio,
            return_text_tokens=self.args.return_text_tokens,
            return_audio_tokens=self.args.return_audio_tokens,
            return_prompt_audio_tokens=self.args.return_prompt_audio_tokens,
            return_cuts=self.args.return_cuts,
            audio_blank_id=self.args.audio_blank_id,
            text_blank_id=self.args.text_blank_id,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                drop_last=self.args.drop_last,
            )

        logging.debug("About to create test dataloader")
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    def test_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        logging.info("About to create test dataset")
        dataset = SpeechSynthesisDataset(
            return_text=self.args.return_text,
            return_audio=self.args.return_audio,
            return_prompt_audio=self.args.return_prompt_audio,
            return_text_tokens=self.args.return_text_tokens,
            return_audio_tokens=self.args.return_audio_tokens,
            return_prompt_audio_tokens=self.args.return_prompt_audio_tokens,
            return_cuts=self.args.return_cuts,
            audio_blank_id=self.args.audio_blank_id,
            text_blank_id=self.args.text_blank_id,
        )
        sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        logging.debug("About to create valid dataloader")
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )
        return dataloader

    @lru_cache()
    def train_small_cuts(self) -> CutSet:
        logging.info("About to get small subset cuts")
        filename = f"libriheavy_cuts_small_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)

    @lru_cache()
    def train_medium_cuts(self) -> CutSet:
        logging.info("About to get medium subset cuts")
        filename = f"libriheavy_cuts_medium_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)

    @lru_cache()
    def train_large_cuts(self) -> CutSet:
        logging.info("About to get large subset cuts")
        filename = f"libriheavy_cuts_large_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        filename = f"libriheavy_cuts_dev_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get the test-clean cuts")
        filename = f"libriheavy_cuts_test_clean_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get the test-other cuts")
        filename = f"libriheavy_cuts_test_other_prompt_{self.args.prompt_duration}s_phone.jsonl.gz"
        return load_manifest_lazy(self.args.manifest_dir / filename)
