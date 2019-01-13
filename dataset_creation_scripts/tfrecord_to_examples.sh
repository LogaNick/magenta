#!/bin/bash

if [ $# -lt 2 ]
then
  echo "Usage: $0 <in dir> <out dir>"
  exit 1
fi

NOTESEQUENCES_FILE=$1
OUTPUT_DIRECTORY=$2
CONFIG=dataset_conditioned_performance_with_dynamics

cd ~/magenta
source activate magenta

nohup ./bazel-bin/magenta/models/performance_rnn/performance_rnn_create_dataset \
--config=${CONFIG} \
--input=$NOTESEQUENCES_FILE \
--output_dir=$OUTPUT_DIRECTORY \
--eval_ratio=0.10
