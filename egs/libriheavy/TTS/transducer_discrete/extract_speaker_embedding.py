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
import logging
import numpy as np
import os
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Union

from speech_enc_datamodule import SpeechEncDataModule
from icefall.dist import cleanup_dist, setup_dist
from icefall.utils import setup_logger
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut, MonoCut
from lhotse.features.io import NumpyHdf5Writer
from lhotse.serialization import SequentialJsonlWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import nemo.collections.asr as nemo_asr


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="small",
        help="""Subset to process. Possible values are 'small', 'medium', 'large',
        'dev', 'test_clean' and 'test_other'""",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/upper_no_punc/manifests_with_4_codebooks"),
        help="Path to directory to save the cuts with encoded codebooks.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/"),
        help="Path to directory to save logs.",
    )

    parser.add_argument(
        "--prompt-duration",
        type=float,
        default=10.0,
        help="Duration in seconds of acoustic prompt.",
    )

    return parser


def encode_dataset(
    dl: torch.utils.data.DataLoader,
    args: argparse.ArgumentParser,
    model: Union[nn.Module, DDP],
    cuts_writer: SequentialJsonlWriter,
    spk_emb_writer: NumpyHdf5Writer,
    device: torch.device,
) -> None:
    """Encode speech dataset and store the recognition results to manifest.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      args:
        The return value of get_parser().parse_args()
      model:
        The speaker encoder model used to extract speaker embeddings.
      cuts_writer:
        Writer to save the cuts with encoded codebooks.
      spk_emb_writer:
        Whiter to save the speaker embedding numpy arrays.
    """
    #  Background worker to save codebooks and cuts to disk.
    def _save_worker(cuts: List[Cut], embeddings: np.ndarray):
        assert len(cuts) == embeddings.shape[0], (len(cuts), embeddings.shape[0])
        for idx in range(len(cuts)):
            cut_data = cuts[idx].to_dict()
            new_cut = MonoCut.from_dict(cut_data)
            # assert hasattr(new_cut, "prompt_cut")
            new_cut.prompt_speaker_embedding = spk_emb_writer.store_array(
                key=new_cut.id, value=embeddings[idx]
            )
            cuts_writer.write(new_cut, flush=True)

    num_cuts = 0
    log_interval = 10
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.
        for batch_idx, batch in enumerate(dl):
            prompt_audio = batch["prompt_audio"].to(device)
            prompt_audio_lens = batch["prompt_audio_lens"].to(device)
            cuts = batch["cut"]

            _, embeddings = model(
                input_signal=prompt_audio, input_signal_length=prompt_audio_lens
            )
            assert embeddings.shape == (
                prompt_audio.shape[0], model.cfg.decoder.emb_sizes
            ), embeddings.shape

            embeddings = embeddings.cpu().numpy()
            futures.append(executor.submit(_save_worker, cuts, embeddings))

            num_cuts += len(cuts)
            if batch_idx % log_interval == 0:
                logging.info(f"cuts processed until now is {num_cuts}")

        for f in futures:
            f.result()


@torch.no_grad()
def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    if world_size > 1:
        setup_dist(rank, world_size, args.master_port)

    setup_logger(f"{args.log_dir}/log-extract-speaker-embedding-{args.prompt_duration}s")
    logging.info("Speaker embedding extraction started")

    logging.info(f"{vars(args)}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"device: {device}")

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name='titanet_large'
    )
    model.to(device)
    model.eval()

    assert args.prompt_sampling_rate == model.cfg.train_ds.sample_rate, (
        args.prompt_sampling_rate, model.cfg.train_ds.sample_rate
    )

    if world_size > 1:
        out_cuts_filename = args.manifest_out_dir / (
            f"{args.out_cuts_filename}_job_{rank}" + args.suffix
        )
        spk_emb_filename = args.manifest_out_dir / (
            f"{args.spk_emb_filename}_job_{rank}"
        )
    else:
        out_cuts_filename = args.manifest_out_dir / (
            f"{args.out_cuts_filename}" + args.suffix
        )
        spk_emb_filename = args.manifest_out_dir / f"{args.spk_emb_filename}"

    # we will store new cuts with encoded codebooks.
    args.return_cuts = True
    args.return_audio = False
    args.return_prompt_audio = True
    data_module = SpeechEncDataModule(args)
    in_cuts = load_manifest_lazy(
        args.manifest_out_dir / (args.in_cuts_filename + args.suffix)
    )
    dl = data_module.dataloader(in_cuts)

    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)
    spk_emb_writer = NumpyHdf5Writer(spk_emb_filename)

    encode_dataset(
        dl=dl,
        args=args,
        model=model,
        cuts_writer=cuts_writer,
        spk_emb_writer=spk_emb_writer,
        device=device,
    )

    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")
    spk_emb_writer.close()
    logging.info(f"Codebooks saved to {spk_emb_filename}")

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    SpeechEncDataModule.add_arguments(parser)
    args = parser.parse_args()

    subset = args.subset
    assert subset in ["small", "medium", "large", "dev", "test_clean", "test_other"], subset

    manifest_out_dir = args.manifest_out_dir

    args.suffix = ".jsonl.gz"
    args.in_cuts_filename = f"libriheavy_cuts_{subset}_{args.prompt_duration}s_prompt"
    args.out_cuts_filename = f"libriheavy_cuts_{subset}_{args.prompt_duration}s_prompt_spk_emb"
    args.spk_emb_filename = f"libriheavy_prompt_spk_emb_{subset}"

    out_cuts_filename = manifest_out_dir / (args.out_cuts_filename + args.suffix)
    if out_cuts_filename.is_file():
        print(f"{out_cuts_filename} already exists - skipping.")
        return

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)

        files_to_combine = f"{manifest_out_dir}/{args.out_cuts_filename}_job_*{args.suffix}"
        os.system(f"lhotse combine {files_to_combine} {out_cuts_filename}")
        print(f"Combined to {out_cuts_filename}")

        os.system(f"rm {files_to_combine}")
    else:
        run(rank=0, world_size=world_size, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
