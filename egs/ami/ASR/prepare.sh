#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100
use_gss=true  # Use GSS-based enhancement with MDM setting

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/amicorpus
#      You can find audio and transcripts in this path.
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#
#  - $dl_dir/{LDC2004S13,LDC2005S13,LDC2004T19,LDC2005T19}
#      These contain the Fisher English audio and transcripts. We will
#      only use the transcripts as extra LM training data (similar to Kaldi).
#
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data
vocab_size=500

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/amicorpus,
  # you can create a symlink
  #
  #   ln -sfv /path/to/amicorpus $dl_dir/amicorpus
  #
  if [ ! -d $dl_dir/amicorpus ]; then
    lhotse download ami --mic ihm $dl_dir/amicorpus
    lhotse download ami --mic mdm $dl_dir/amicorpus
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare AMI manifests"
  # We assume that you have downloaded the AMI corpus
  # to $dl_dir/amicorpus. We perform text normalization for the transcripts.
  mkdir -p data/manifests
  for mic in ihm sdm mdm; do
    lhotse prepare ami --mic $mic --partition full-corpus-asr --normalize-text kaldi \
      --max-words-per-segment 30 $dl_dir/amicorpus data/manifests/
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to $dl_dir/musan
  mkdir -p data/manifests
  lhotse prepare musan $dl_dir/musan data/manifests
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] && [ $use_gss = true ]; then
  log "Stage 3: Apply GSS enhancement on MDM data (this stage requires a GPU)"
  # We assume that you have installed the GSS package: https://github.com/desh2608/gss
  local/prepare_ami_gss.sh data/manifests exp/ami_gss
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank features for AMI"
  mkdir -p data/fbank
  python local/compute_fbank_ami.py
  log "Combine features from train splits"
  lhotse combine data/manifests/cuts_train_{ihm,ihm_rvb,sdm,gss}.jsonl.gz - | shuf |\
    gzip -c > data/manifests/cuts_train_all.jsonl.gz
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank features for musan"
  mkdir -p data/fbank
  python local/compute_fbank_musan.py
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Dump transcripts for BPE model training."
  mkdir -p data/lm
  cat <(gunzip -c data/manifests/ami-sdm_supervisions_train.jsonl.gz | jq '.text' | sed 's:"::g')> data/lm/transcript_words.txt
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare BPE based lang"

  lang_dir=data/lang_bpe_${vocab_size}
  mkdir -p $lang_dir

  # Add special words to words.txt
  echo "<eps> 0" > $lang_dir/words.txt
  echo "!SIL 1" >> $lang_dir/words.txt
  echo "<UNK> 2" >> $lang_dir/words.txt

  # Add regular words to words.txt
  cat data/lm/transcript_words.txt | grep -o -E '\w+' | sort -u | awk '{print $0,NR+2}' >> $lang_dir/words.txt

  # Add remaining special word symbols expected by LM scripts.
  num_words=$(cat $lang_dir/words.txt | wc -l)
  echo "<s> ${num_words}" >> $lang_dir/words.txt
  num_words=$(cat $lang_dir/words.txt | wc -l)
  echo "</s> ${num_words}" >> $lang_dir/words.txt
  num_words=$(cat $lang_dir/words.txt | wc -l)
  echo "#0 ${num_words}" >> $lang_dir/words.txt

  ./local/train_bpe_model.py \
    --lang-dir $lang_dir \
    --vocab-size $vocab_size \
    --transcript data/lm/transcript_words.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang_bpe.py --lang-dir $lang_dir
  fi
fi
