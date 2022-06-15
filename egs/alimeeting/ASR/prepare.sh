#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/alimeeting
#     This directory contains the following files downloaded from
#       https://openslr.org/62/
#
#     - Train_Ali_far.tar.gz
#     - Train_Ali_near.tar.gz
#     - Test_Ali.tar.gz
#     - Eval_Ali.tar.gz
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

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

  if [ ! -f $dl_dir/alimeeting/Train_Ali_far.tar.gz ]; then
    lhotse download ali-meeting $dl_dir/alimeeting
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare alimeeting manifest"
  # We assume that you have downloaded the alimeeting corpus
  # to $dl_dir/alimeeting
  if [ ! -f data/manifests/alimeeting/.manifests.done ]; then
    mkdir -p data/manifests/alimeeting
    lhotse prepare ali-meeting $dl_dir/alimeeting data/manifests/alimeeting
    touch data/manifests/alimeeting/.manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Process alimeeting"
  if [ ! -f data/fbank/alimeeting/.fbank.done ]; then
    mkdir -p data/fbank/alimeeting
    lhotse prepare ali-meeting $dl_dir/alimeeting data/manifests/alimeeting
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for alimeeting"
  if [ ! -f data/fbank/.alimeeting.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_alimeeting.py
    touch data/fbank/.alimeeting.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare char based lang"
  lang_char_dir=data/lang_char
  mkdir -p $lang_char_dir

  # Prepare text.
  # Note: in Linux, you can install jq with the  following command:
  # wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
  gunzip -c data/manifests/alimeeting/supervisions_train.jsonl.gz \
    | jq ".text" | sed 's/"//g' \
    | ./local/text2token.py -t "char" > $lang_char_dir/text

  # Prepare words segments
  python ./local/text2segments.py \
    --input $lang_char_dir/text \
    --output $lang_char_dir/text_words_segmentation

  cat $lang_char_dir/text_words_segmentation | sed "s/ /\n/g" \
    | sort -u | sed "/^$/d" \
    | uniq > $lang_char_dir/words_no_ids.txt

  # Prepare words.txt
  if [ ! -f $lang_char_dir/words.txt ]; then
    ./local/prepare_words.py \
      --input-file $lang_char_dir/words_no_ids.txt \
      --output-file $lang_char_dir/words.txt
  fi

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py
  fi
fi
