#!/bin/bash

if [ $# -lt 2 ]
then
  echo "Usage: $0 <in dir> <out dir>"
  exit 1
fi

INPUT_DIRECTORY=$1
SEQUENCES_TFRECORD=$2

cd ~/magenta
source activate magenta

./bazel-bin/magenta/scripts/convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
