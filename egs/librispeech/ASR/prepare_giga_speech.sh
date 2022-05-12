#!/usr/bin/env bash

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/GigaSpeech
#      You can find audio, dict, GigaSpeech.json inside it.
#      You can apply for the download credentials by following
#      https://github.com/SpeechColab/GigaSpeech#download

# Number of hours for GigaSpeech subsets
# XL 10k hours
# L  2.5k hours
# M  1k hours
# S  250 hours
# XS 10 hours
# DEV 12 hours
# Test 40 hours

# Split XL subset to this number of pieces
# This is to avoid OOM during feature extraction.
num_splits=2000
# We use lazy split from lhotse.
# The XL subset (10k hours) contains 37956 cuts without speed perturbing.
# We want to split it into 2000 splits, so each split
# contains about 37956 / 2000 = 19 cuts. As a result, there will be 1998 splits.
chunk_size=19 # number of cuts in each split. The last split may contain fewer cuts.

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  [ ! -e $dl_dir/GigaSpeech ] && mkdir -p $dl_dir/GigaSpeech

  # If you have pre-downloaded it to /path/to/GigaSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/GigaSpeech $dl_dir/GigaSpeech
  #
  if [ ! -d $dl_dir/GigaSpeech/audio ] && [ ! -f $dl_dir/GigaSpeech.json ]; then
    # Check credentials.
    if [ ! -f $dl_dir/password ]; then
      echo -n "$0: Please apply for the download credentials by following"
      echo -n "https://github.com/SpeechColab/GigaSpeech#dataset-download"
      echo " and save it to $dl_dir/password."
      exit 1;
    fi
    PASSWORD=`cat $dl_dir/password 2>/dev/null`
    if [ -z "$PASSWORD" ]; then
      echo "$0: Error, $dl_dir/password is empty."
      exit 1;
    fi
    PASSWORD_MD5=`echo $PASSWORD | md5sum | cut -d ' ' -f 1`
    if [[ $PASSWORD_MD5 != "dfbf0cde1a3ce23749d8d81e492741b8" ]]; then
      echo "$0: Error, invalid $dl_dir/password."
      exit 1;
    fi
    # Download XL, DEV and TEST sets by default.
    lhotse download gigaspeech \
      --subset XL \
      --subset L \
      --subset M \
      --subset S \
      --subset XS \
      --subset DEV \
      --subset TEST \
      --host tsinghua \
      $dl_dir/password $dl_dir/GigaSpeech
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare GigaSpeech manifest (may take 30 minutes)"
  # We assume that you have downloaded the GigaSpeech corpus
  # to $dl_dir/GigaSpeech
  mkdir -p data/manifests
  lhotse prepare gigaspeech \
    --subset XL \
    --subset L \
    --subset M \
    --subset S \
    --subset XS \
    --subset DEV \
    --subset TEST \
    -j $nj \
    $dl_dir/GigaSpeech data/manifests
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Preprocess GigaSpeech manifest"
  if [ ! -f data/fbank/.preprocess_complete ]; then
   log "It may take 2 hours for this stage"
   python3 ./local/preprocess_gigaspeech.py
   touch data/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute features for DEV and TEST subsets of GigaSpeech (may take 2 minutes)"
  python3 ./local/compute_fbank_gigaspeech_dev_test.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Split XL subset into ${num_splits} pieces"
  split_dir=data/fbank/XL_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split-lazy ./data/fbank/cuts_XL_raw.jsonl.gz $split_dir $chunk_size
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute features for XL"
  # Note: The script supports --start and --stop options.
  # You can use several machines to compute the features in parallel.
  python3 ./local/compute_fbank_gigaspeech_splits.py \
    --num-workers $nj \
    --batch-duration 600 \
    --num-splits $num_splits
fi
