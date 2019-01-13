#!/bin/bash

if [ $# -lt 3 ]
then
  echo "Usage: $0 <in dir> <out dir> <file number> <config>"
  exit 1
fi

NOTESEQUENCES_FILE=$1
OUTPUT_DIRECTORY=$2
FILE_NUM=$3
CONFIG=$4

cd ~/magenta
source activate magenta

nohup ./bazel-bin/magenta/models/performance_rnn/performance_rnn_create_dataset \
--config=${CONFIG} \
--input=$NOTESEQUENCES_FILE \
--output_dir=$OUTPUT_DIRECTORY \
--eval_ratio=0.10 \
--num_threads=$FILE_NUM > nohup_sequence_ex.out
