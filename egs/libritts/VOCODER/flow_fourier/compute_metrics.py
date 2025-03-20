# Copyright      2024  Xiaomi Corporation     (Author: Zengwei Yao)
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

# Modified from
# - https://github.com/bfs18/rfwave/blob/main/calculate_voc_metrics.py
# - https://github.com/gemelo-ai/vocos/blob/main/vocos/experiment.py


import argparse
import logging
from pathlib import Path

import torch
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from icefall.utils import setup_logger, str2bool
from pesq import pesq
from torch import nn

from audio_utils import safe_log
from periodicity import calculate_periodicity_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-wav-dir", type=Path, help="Directory to the ground-truth wav files"
    )
    parser.add_argument(
        "--pred-wav-dir", type=Path, help="Directory to the predicted wav files"
    )
    parser.add_argument(
        "--wav-list-file", type=Path, help="wav list file of test set."
    )
    parser.add_argument(
        "--use-periodicity",
        type=str2bool,
        default=False,
        help="Whether to compute periodicity metrics"
    )
    return parser.parse_args()


def mel_loss_fun(mel_fun: nn.Module, gt_wav: torch.Tensor, pred_wav: torch.Tensor):
    gt_mel = mel_fun(gt_wav)
    pred_mel = mel_fun(pred_wav)
    return (safe_log(gt_mel) - safe_log(pred_mel)).abs().mean().item()


def compute_metrics(gt_wav_dir: Path, pred_wav_dir: Path, wav_list_file: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mrstft_fun = MultiResolutionSTFTLoss(device=device)

    sampling_rate = 24000
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    mel_fun = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        center=True,
        power=2,
    )
    mel_fun.to(device)

    tot_mrstft_loss = 0
    tot_mel_loss = 0
    tot_pesq_score = 0
    tot_periodicity_loss = 0
    tot_pitch_loss = 0
    tot_f1_score = 0
    cnt = 0
    with open(wav_list_file) as f:
        wav_list = f.read().splitlines()
    for fname in wav_list:
        gt_wav_file = gt_wav_dir / Path(fname)
        pred_wav_file = pred_wav_dir / Path(fname)
        cnt += 1
        # load audio
        gt_wav, sr = torchaudio.load(str(gt_wav_file))
        assert sr == sampling_rate
        pred_wav, sr = torchaudio.load(str(pred_wav_file))
        assert sr == sampling_rate

        # trim to equal length
        min_len = min(gt_wav.shape[1], pred_wav.shape[1])
        gt_wav = gt_wav[0, :min_len].to(device)  # (T)
        pred_wav = pred_wav[0, :min_len].to(device)  # (T)

        # multi-resolution STFT loss
        mrstft_loss = mrstft_fun(pred_wav[None, None], gt_wav[None, None])
        tot_mrstft_loss += mrstft_loss

        # log-mel-spectrogram loss
        mel_loss = mel_loss_fun(mel_fun, gt_wav, pred_wav)
        tot_mel_loss += mel_loss

        gt_wav_16k = torchaudio.functional.resample(gt_wav, orig_freq=sr, new_freq=16000)
        pred_wav_16k = torchaudio.functional.resample(pred_wav, orig_freq=sr, new_freq=16000)

        # pesq
        pesq_score = pesq(16000, gt_wav_16k.cpu().numpy(), pred_wav_16k.cpu().numpy(), 'wb', on_error=1)
        tot_pesq_score += pesq_score

        if args.use_periodicity:
            # periodicity, pitch, f1
            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(
                gt_wav_16k.unsqueeze(0), pred_wav_16k.unsqueeze(0)
            )
            tot_periodicity_loss += periodicity_loss
            tot_pitch_loss += pitch_loss
            tot_f1_score += f1_score

        if cnt % 100 == 0:
            logging.info(f"Processed {cnt} samples...")

    logging.info(f"Number of samples: {cnt}")
    logging.info(f"mrstft loss: {(tot_mrstft_loss / cnt):.2f}")
    logging.info(f"mel loss: {(tot_mel_loss / cnt):.2f}")
    logging.info(f"pesq score: {(tot_pesq_score / cnt):.2f}")
    if args.use_periodicity:
        logging.info(f"periodicity loss: {(tot_periodicity_loss / cnt):.2f}")
        logging.info(f"pitch loss: {(tot_pitch_loss / cnt):.2f}")
        logging.info(f"f1 score: {(tot_f1_score / cnt):.2f}")
    logging.info("Done!")


if __name__ == "__main__":
    args = get_args()
    setup_logger(f"{args.pred_wav_dir}/log_metrics")

    logging.info(f"Start computing metrics for pred_wav_dir={args.pred_wav_dir}, wav_list_file={args.wav_list_file}")
    compute_metrics(args.gt_wav_dir, args.pred_wav_dir, args.wav_list_file)
