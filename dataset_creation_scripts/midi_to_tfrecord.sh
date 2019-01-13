#!/bin/bash

if [ $# -lt 3 ]
then
  echo "Usage: $0 <in dir> <out dir> <file_num> <config>"
  exit 1
fi

INPUT_DIRECTORY=$1
SEQUENCES_TFRECORD=$2
FILE_NUM=$3
CONFIG=$4

cd ~/magenta
source activate magenta

nohup ./bazel-bin/magenta/scripts/convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --config=$CONFIG \
  --num_threads=$FILE_NUM > nohup_note_sequences.out
