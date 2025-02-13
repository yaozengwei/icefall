#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
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
import os

import soundfile as sf
import torch
import torch.nn as nn
from checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from dataset import build_data_loader
from icefall.utils import AttributeDict, setup_logger, str2bool
from train_one_step import add_model_arguments, get_model


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=100,
        help="""Number of checkpoints to average. Automatically select
        consecutive checkpoints before the checkpoint specified by --epoch""",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="flow_fourier/exp",
        help="The experiment dir.",
    )

    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="download/libritts/LibriTTS",
        help="Root dir to the LibriTTS dataset.",
    )

    parser.add_argument(
        "--test-wav-list",
        type=str,
        default="data/wav_list/test.txt",
        help="Wav list file of test set.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="How many subprocesses to use for data loading.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing inference parameters.
    """
    params = AttributeDict(
        {
            "sampling_rate": 24000,
        }
    )

    return params


def infer_audio(
    params: AttributeDict,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run the inference process."""
    def makedir_if_necessary(filename: str):
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    device = next(model.parameters()).device
    total_samples = len(dataloader.dataset)
    cnt = 0
    log_interval = 10
    with torch.inference_mode():
        for batch_idx, (audios, audio_lens, file_names) in enumerate(dataloader):
            batch_size = audios.shape[0]
            audios = audios.to(device)  # (batch, time)
            audio_lens = audio_lens.to(device)  # (batch,)
            pred_audios = model.infer(audio=audios, audio_lens=audio_lens)
            for i in range(batch_size):
                pred = pred_audios[i, :audio_lens[i].item()].data.cpu().numpy()
                pred_out_file = f"{params.wav_dir_pred}/{file_names[i]}"
                makedir_if_necessary(pred_out_file)
                sf.write(pred_out_file, pred, params.sampling_rate)

            cnt += batch_size
            if batch_idx % log_interval == 0:
                logging.info(f"Processed {cnt} / {total_samples} samples")

        logging.info(f"Processed {cnt} samples in total.")


def main():
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-infer")
    logging.info("Inference start")

    params.suffix = f"wav-epoch-{params.epoch}-avg-{params.avg}"
    if params.use_averaged_model:
        params.suffix += "-use-avg-model"

    params.wav_dir_pred = f"{params.exp_dir}/{params.suffix}-pred"
    os.makedirs(params.wav_dir_pred, exist_ok=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_param}")

    if not params.use_averaged_model:
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {params.epoch}"
        )
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )

    model.to(device)
    model.eval()

    # assert params.batch_size == 1, "Currently only support inference with batch_size=1"
    dataloader = build_data_loader(
        wav_list_file=params.test_wav_list,
        corpus_dir=params.corpus_dir,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        train=False,
        drop_last=False,
    )

    infer_audio(params=params, model=model, dataloader=dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    main()
