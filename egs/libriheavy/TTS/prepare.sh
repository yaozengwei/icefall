#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export CUDA_VISIBLE_DEVICES="2"
world_size=1

set -eou pipefail

stage=1
stop_stage=100

encodec_model_dir=data/encodec_model

# Path to save the manifests
output_dir=$PWD/data/cases_and_punc

prompt_duration=3

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Perform speech encoding"
  # for subset in dev test_clean test_other small medium large; do
  for subset in test_clean test_other small; do
    ./transducer_discrete/encode_speech.py \
      --world-size $world_size \
      --num-workers 24 \
      --subset $subset \
      --manifest-in-dir $output_dir/manifests \
      --manifest-out-dir $output_dir/manifests_codebooks \
      --encodec-model-dir $encodec_model_dir \
      --max-duration 1000 \
      --master-port 12345
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare acoutic prompts "
  for subset in dev test_clean test_other small; do
    ./transducer_discrete/prepare_prompt.py \
      --subset $subset \
      --manifest-dir $output_dir/manifests_codebooks \
      --prompt-duration $prompt_duration \
      --num-jobs 10
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Convert text to phonemes."
  for subset in dev test_clean test_other small; do
    ./transducer_discrete/prepare_phonemes.py \
      --subset $subset \
      --manifest-dir $output_dir/manifests_codebooks \
      --prompt-duration $prompt_duration
  done
fi

