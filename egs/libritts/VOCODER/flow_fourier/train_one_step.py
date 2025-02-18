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
import copy
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.utils import fix_random_seed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from optim import ScaledAdam, Eden
from torch.utils.tensorboard import SummaryWriter
from utils import MetricsTracker, plot_feature
from model_one_step import OneStepVocoder
from dataset import build_data_loader

from icefall import diagnostics
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, setup_logger, str2bool
from checkpoint import (
    load_checkpoint,
    save_checkpoint,
    update_averaged_model,
)
LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


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
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="flow_fourier/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.04, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=1,
        help="""Save checkpoint after processing this number of epochs"
        periodically. We save checkpoint to exp-dir/ whenever
        params.cur_epoch % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/epoch-{params.cur_epoch}.pt'.
        Since it will take around 1000 epochs, we suggest using a large
        save_every_n to save disk space.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--train-wav-list",
        type=str,
        default="data/wav_list/train-full-960.txt",
        help="Wav list file of training set.",
    )

    parser.add_argument(
        "--valid-wav-list",
        type=str,
        default="data/wav_list/validation.txt",
        help="Wav list file of validation set.",
    )

    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="download/libritts/LibriTTS",
        help="Root dir to the LibriTTS dataset.",
    )

    parser.add_argument(
        "--inv-noise-dir",
        type=str,
        default="",
        help="Dir to the saved inverted noise.",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=32768,
        help="",
    )

    parser.add_argument(
        "--valid-num-samples",
        type=int,
        default=32768,
        help="",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="How many subprocesses to use for data loading.",
    )

    parser.add_argument(
        "--kaldi-io",
        type=str2bool,
        default=False,
        help="Whether to use kaldi_native_io style dataset",
    )

    parser.add_argument(
        "--train-scp-ark-dir",
        type=str,
        default="./data/scp_ark/train-full-960",
        help="Dir to the scp/ark files of the training set.",
    )

    parser.add_argument(
        "--valid-scp-ark-dir",
        type=str,
        default="./data/scp_ark/validation",
        help="Dir to the scp/ark files of the validation set.",
    )

    parser.add_argument(
        "--train-inv-noise-scp-ark-dir",
        type=str,
        default="",
        help="Dir to the scp/ark files of the inv-noise training set.",
    )

    parser.add_argument(
        "--valid-inv-noise-scp-ark-dir",
        type=str,
        default="",
        help="Dir to the scp/ark files of the inv-noise validation set.",
    )

    parser.add_argument(
        "--mel-scaling-loss",
        type=str2bool,
        default=True,
        help="",
    )

    parser.add_argument(
        "--mix-noise-scale",
        type=float,
        default=0.2,
        help="",
    )

    parser.add_argument(
        "--disc-loss-scale",
        type=float,
        default=1.0,
        help="",
    )

    parser.add_argument(
        "--log-mel-loss-scale",
        type=float,
        default=1.0,
        help="",
    )

    parser.add_argument(
        "--fft-mag-loss-scale",
        type=float,
        default=1.0,
        help="",
    )

    add_model_arguments(parser)

    return parser


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="",
    )

    parser.add_argument(
        "--n-ffts",
        type=str,
        default="512,256,128",
        help="",
    )

    parser.add_argument(
        "--hop-lengths",
        type=str,
        default="256,128,64",
        help="",
    )

    parser.add_argument(
        "--mel-n-fft",
        type=int,
        default=512,
        help="",
    )

    parser.add_argument(
        "--mel-hop-length",
        type=int,
        default=256,
        help="",
    )

    parser.add_argument(
        "--convnext-num-layers",
        type=str,
        default="8,8,8",
        help="",
    )

    parser.add_argument(
        "--convnext-channels",
        type=str,
        default="512,512,512",
        help="",
    )

    parser.add_argument(
        "--mel-enc-channels",
        type=int,
        default=512,
        help="",
    )

    parser.add_argument(
        "--mel-enc-num-layers",
        type=int,
        default=4,
        help="",
    )

    parser.add_argument(
        "--from-inv-mel",
        type=str2bool,
        default=True,
        help="Whether to construct x0 from inverted Mel-spectrogram with random phase.",
    )

    parser.add_argument(
        "--init-noise-scale",
        type=float,
        default=0.1,
        help="Noise scale used when constructing x0 from standard distribution.",
    )

    parser.add_argument(
        "--use-disc-loss",
        type=str2bool,
        default=False,
        help="Whether to discriminator loss",
    )

    parser.add_argument(
        "--use-fft-mag-loss",
        type=str2bool,
        default=False,
        help="Whether to fft magnitude l1-loss",
    )

    parser.add_argument(
        "--use-log-mel-loss",
        type=str2bool,
        default=False,
        help="Whether to log-mel l1-loss",
    )


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.
    """
    params = AttributeDict(
        {
            # training params
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": -1,  # 0
            "log_interval": 50,
            "valid_interval": 300,
            "env_info": get_env_info(),
            "sampling_rate": 24000,
            "branch_drop_rate": 0.1,
            "warm_step": 2000,
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch - 1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_model(params: AttributeDict) -> nn.Module:
    model = OneStepVocoder(
        n_mels=params.n_mels,
        sampling_rate=params.sampling_rate,
        n_ffts=_to_int_tuple(params.n_ffts),
        hop_lengths=_to_int_tuple(params.hop_lengths),
        mel_n_fft=params.mel_n_fft,
        mel_hop_length=params.mel_hop_length,
        mel_enc_channels=params.mel_enc_channels,
        mel_enc_num_layers=params.mel_enc_num_layers,
        convnext_num_layers=_to_int_tuple(params.convnext_num_layers),
        convnext_channels=_to_int_tuple(params.convnext_channels),
        from_inv_mel=params.from_inv_mel,
        init_noise_scale=params.init_noise_scale,
        use_disc_loss=params.use_disc_loss,
        use_fft_mag_loss=params.use_fft_mag_loss,
        use_log_mel_loss=params.use_log_mel_loss,
    )
    return model


def compute_loss(
    audio: torch.Tensor,
    audio_lens: torch.Tensor,
    inv_noise: torch.Tensor,
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    is_training: bool = True,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute loss given the model and its inputs."""
    # linearly decrease from branch_drop_rate to 0 in first warm_step batches
    branch_drop_rate = params.branch_drop_rate * (1.0 - params.batch_idx_train / params.warm_step)
    branch_drop_rate = max(branch_drop_rate, 0.0)

    use_disc_loss = params.use_disc_loss
    use_log_mel_loss = params.use_log_mel_loss
    use_fft_mag_loss = params.use_fft_mag_loss

    with torch.set_grad_enabled(is_training):
        losses = model(
            audio=audio,
            audio_lens=audio_lens,
            inv_noise=inv_noise,
            mel_scaling_loss=params.mel_scaling_loss,
            branch_drop_rate=branch_drop_rate,
            mix_noise_scale=params.mix_noise_scale,
        )
        main_loss, log_mel_loss, fft_mag_loss, disc_loss = losses
        loss = main_loss * 1.0
        if use_log_mel_loss:
            loss += params.log_mel_loss_scale * log_mel_loss
        if use_fft_mag_loss:
            loss += params.fft_mag_loss_scale * fft_mag_loss
        if use_disc_loss:
            loss += params.disc_loss_scale * disc_loss

    assert loss.requires_grad == is_training

    batch_size = audio.shape[0]
    loss_info = MetricsTracker()
    loss_info["samples"] = batch_size
    loss_info["loss"] = loss.detach().item() * batch_size
    loss_info["main_loss"] = main_loss.detach().item() * batch_size
    if use_log_mel_loss:
        loss_info["log_mel_loss"] = log_mel_loss.detach().item() * batch_size
    if use_fft_mag_loss:
        loss_info["fft_mag_loss"] = fft_mag_loss.detach().item() * batch_size
    if use_disc_loss:
        loss_info["disc_loss"] = disc_loss.detach().item() * batch_size

    return loss, loss_info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            # sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1

        audio = batch[0].to(device)
        audio_lens = torch.full((audio.shape[0],), audio.shape[1], dtype=torch.int32, device=device)
        inv_noise = batch[1].to(device)
        # audio: (N, T), float32
        # audio_lens, (N,), int32
        batch_size = audio.shape[0]

        try:
            with autocast(enabled=params.use_fp16):
                # forward discriminator
                loss, loss_info = compute_loss(
                    audio=audio,
                    audio_lens=audio_lens,
                    inv_noise=inv_noise,
                    params=params,
                    model=model,
                    is_training=True,
                )

            # summary stats
            tot_loss = tot_loss + loss_info

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        # Update model_avg at every average_period steps
        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if params.use_fp16:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info, infer_sample = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
                rank=rank,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated() // 1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )
                pred_from_rand, pred_from_inv, gt_audio = infer_sample

                tb_writer.add_audio(
                    "train/valid_pred_audio_from_random_noise",
                    pred_from_rand,
                    params.batch_idx_train,
                    params.sampling_rate,
                )
                tb_writer.add_audio(
                    "train/valid_pred_audio_from_inv_noise",
                    pred_from_inv,
                    params.batch_idx_train,
                    params.sampling_rate,
                )
                tb_writer.add_audio(
                    "train/valid_gt_audio",
                    gt_audio,
                    params.batch_idx_train,
                    params.sampling_rate,
                )

                def compute_spec(y):
                    stft = librosa.stft(y, n_fft=1024)
                    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                tb_writer.add_image(
                    "train/valid_pred_audio_from_random_noise_spec",
                    plot_feature(compute_spec(pred_from_rand)),
                    params.batch_idx_train,
                    dataformats="HWC",
                )
                tb_writer.add_image(
                    "train/valid_pred_audio_from_inv_noise_spec",
                    plot_feature(compute_spec(pred_from_inv)),
                    params.batch_idx_train,
                    dataformats="HWC",
                )
                tb_writer.add_image(
                    "train/valid_gt_audio_spec",
                    plot_feature(compute_spec(gt_audio)),
                    params.batch_idx_train,
                    dataformats="HWC",
                )

    loss_value = tot_loss["loss"] / tot_loss["samples"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[MetricsTracker, Tuple[np.ndarray, np.ndarray]]:
    """Run the validation process."""
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    # used to summary the stats over iterations
    tot_loss = MetricsTracker()
    returned_sample = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            audio = batch[0].to(device)
            audio_lens = torch.full((audio.shape[0],), audio.shape[1], dtype=torch.int32, device=device)
            inv_noise = batch[1].to(device)
            # audio: (N, T), float32
            # audio_lens, (N,), int32

            loss, loss_info = compute_loss(
                audio=audio,
                audio_lens=audio_lens,
                inv_noise=inv_noise,
                params=params,
                model=model,
                is_training=False,
            )
            assert loss.requires_grad is False
            # summary stats
            tot_loss = tot_loss + loss_info

            # infer for first batch:
            if batch_idx == 0 and rank == 0:
                inner_model = model.module if isinstance(model, DDP) else model
                # infer from a random noise
                pred_from_rand = inner_model.infer(
                    audio=audio[:1, :audio_lens[0].item()],
                    audio_lens=audio_lens[:1],
                    log_mel_diff=True,
                )
                pred_from_rand = pred_from_rand[0, :audio_lens[0].item()].data.cpu().numpy()
                # infer from the inverted noise
                pred_from_inv = inner_model.infer(
                    audio=audio[:1, :audio_lens[0].item()],
                    audio_lens=audio_lens[:1],
                    inv_noise=inv_noise[:1, :audio_lens[0].item()],
                    log_mel_diff=True,
                )
                pred_from_inv = pred_from_inv[0, :audio_lens[0].item()].data.cpu().numpy()
                gt_audio = audio[0, :audio_lens[0].item()].data.cpu().numpy()
                returned_sample = (pred_from_rand, pred_from_inv, gt_audio)

    if world_size > 1:
        tot_loss.reduce(device)

    loss_value = tot_loss["loss"] / tot_loss["samples"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss, returned_sample


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0 and not params.print_diagnostics:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params,
        model=model,
        model_avg=model_avg,
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(model.named_parameters(), lr=params.base_lr, clipping_scale=2.0)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs, warmup_start=0.1)

    if checkpoints is not None:
        # load state_dict for optimizer and scheduler
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    train_dl = build_data_loader(
        wav_list_file=params.train_wav_list,
        corpus_dir=params.corpus_dir,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        train=True,
        num_samples=params.train_num_samples,
        world_size=world_size,
        inv_noise_pair=True,
        inv_noise_dir=params.inv_noise_dir,
        kaldi_io=params.kaldi_io,
        scp_ark_dir=params.train_scp_ark_dir,
        inv_noise_scp_ark_dir=params.train_inv_noise_scp_ark_dir,
    )
    valid_dl = build_data_loader(
        wav_list_file=params.valid_wav_list,
        corpus_dir=params.corpus_dir,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        train=False,
        num_samples=params.valid_num_samples,
        world_size=world_size,
        inv_noise_pair=True,
        inv_noise_dir=params.inv_noise_dir,
        kaldi_io=params.kaldi_io,
        scp_ark_dir=params.valid_scp_ark_dir,
        inv_noise_scp_ark_dir=params.valid_inv_noise_scp_ark_dir,
    )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        if world_size > 1:
            # Calling the set_epoch() method on the DistributedSampler
            train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        if epoch % params.save_every_n == 0 or epoch == params.num_epochs:
            filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                # sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            if rank == 0:
                if params.best_train_epoch == params.cur_epoch:
                    best_train_filename = params.exp_dir / "best-train-loss.pt"
                    copyfile(src=filename, dst=best_train_filename)

                if params.best_valid_epoch == params.cur_epoch:
                    best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                    copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
