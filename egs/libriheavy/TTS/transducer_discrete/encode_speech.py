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

from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask
from speech_enc_datamodule import SpeechEncDataModule
from icefall.dist import cleanup_dist, setup_dist
from icefall.utils import AttributeDict, setup_logger
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut, MonoCut
from lhotse.features.io import NumpyHdf5Writer
from lhotse.serialization import SequentialJsonlWriter
from torch.nn.parallel import DistributedDataParallel as DDP


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
        "--manifest-in-dir",
        type=Path,
        default=Path("data/upper_no_punc/manifests"),
        help="Path to directory with cuts.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/upper_no_punc/manifests_codebooks"),
        help="Path to directory to save the cuts with encoded codebooks.",
    )

    parser.add_argument(
        "--encodec-model-dir",
        type=Path,
        default=Path("data/encodec_model/"),
        help="Path to the encodec model directory that contains 'config.yaml' and 'model.pth'.",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log"),
        help="Path to directory to save logs.",
    )

    return parser


def encode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    cuts_writer: SequentialJsonlWriter,
    codebooks_writer: NumpyHdf5Writer,
) -> None:
    """Encode speech dataset and store the recognition results to manifest.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model used to encode speech.
      cuts_writer:
        Writer to save the cuts with encoded codebooks.
      codebooks_writer:
        Whiter to save the encoded codebooks numpy arrays.
    """
    #  Background worker to save codebooks and cuts to disk.
    def _save_worker(cuts: List[Cut], codebooks: np.ndarray, num_frames: np.ndarray):
        assert len(cuts) == codebooks.shape[0], (len(cuts), codebooks.shape[0])
        for idx in range(len(cuts)):
            cut_data = cuts[idx].to_dict()
            # remove "features" field if exists
            cut_data.pop("features", None)

            new_cut = MonoCut.from_dict(cut_data)
            new_cut.codebooks = codebooks_writer.store_array(
                key=new_cut.id,
                value=codebooks[idx][: num_frames[idx]],
                frame_shift=params.encoder_hop_length / params.sampling_rate,
                temporal_dim=0,
                start=new_cut.start,
            )
            cuts_writer.write(new_cut, flush=True)

    num_cuts = 0
    log_interval = 100
    futures = []
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.
        for batch_idx, batch in enumerate(dl):
            audio = batch["audio"].unsqueeze(1).to(device)  # (B, 1, T)
            audio_lens = batch["audio_lens"]
            cuts = batch["cut"]
            num_frames = torch.ceil(audio_lens / params.encoder_hop_length).int().numpy()

            # 1 * n_q x B x T
            codebooks = model.inference_encoding(audio, need_recon=False)["code_indices"]
            assert len(codebooks) == 1, len(codebooks)
            codebooks = codebooks[0].permute(1, 2, 0).cpu().numpy()  # (B, T, n_q)
            assert np.min(codebooks) >= 0, np.min(codebooks)
            assert np.max(codebooks) < params.codebook_size, np.max(codebooks)

            futures.append(executor.submit(_save_worker, cuts, codebooks, num_frames))

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
    params = AttributeDict()
    params.update(vars(args))

    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.log_dir}/log-encode")
    logging.info("Encoding started")
    logging.info(f"{params}")

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
    logging.info(f"device: {device}")

    model, model_args = GANSpeechCodecTask.build_model_from_file(
        config_file=params.encodec_model_dir / "config.yaml",
        model_file=params.encodec_model_dir / "model.pth",
        device=device,
    )
    model.eval()

    assert model_args.sampling_rate == params.sampling_rate, model_args.sampling_rate
    params.encoder_hop_length = model_args.quantizer_conf["encoder_hop_length"]
    params.codebook_size = model_args.quantizer_conf["codebook_size"]

    if world_size > 1:
        out_cuts_filename = params.manifest_out_dir / (f"{params.cuts_filename}_job_{rank}" + params.suffix)
        codebooks_filename = params.manifest_out_dir / f"{params.codebooks_filename}_job_{rank}"
    else:
        out_cuts_filename = params.manifest_out_dir / (f"{params.cuts_filename}" + params.suffix)
        codebooks_filename = params.manifest_out_dir / f"{params.codebooks_filename}"

    # we will store new cuts with encoded codebooks.
    args.return_cuts = True
    data_module = SpeechEncDataModule(args)
    in_cuts = load_manifest_lazy(args.manifest_in_dir / (args.cuts_filename + args.suffix))
    dl = data_module.dataloader(in_cuts)

    cuts_writer = CutSet.open_writer(out_cuts_filename, overwrite=True)
    codebooks_writer = NumpyHdf5Writer(codebooks_filename)

    encode_dataset(
        dl=dl,
        params=params,
        model=model,
        cuts_writer=cuts_writer,
        codebooks_writer=codebooks_writer,
    )

    cuts_writer.close()
    logging.info(f"Cuts saved to {out_cuts_filename}")
    codebooks_writer.close()
    logging.info(f"Codebooks saved to {codebooks_filename}")

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
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    args.suffix = ".jsonl.gz"
    args.cuts_filename = f"libriheavy_cuts_{subset}"
    args.codebooks_filename = f"libriheavy_codebooks_{subset}"

    out_cuts_filename = manifest_out_dir / (args.cuts_filename + args.suffix)
    if out_cuts_filename.is_file():
        print(f"{out_cuts_filename} already exists - skipping.")
        return

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)

        files_to_combine = f"{manifest_out_dir}/{args.cuts_filename}_job_*{args.suffix}"
        os.system(f"lhotse combine {files_to_combine} {out_cuts_filename}")
        print(f"Combined to {out_cuts_filename}")

        os.system(f"rm {files_to_combine}")
    else:
        run(rank=0, world_size=world_size, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
