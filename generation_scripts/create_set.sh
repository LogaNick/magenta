#!/bin/bash

# Script to generate a set of experiments on perfRNN

if (( $# < 5 )); then
	echo "Usage: $0 <num outputs> <run_dir> <output_dir> <config> [<temps to iterate>]"
	exit 1
elif (( $1 < 1 )); then
	echo "Error: Must specify at least 1 output"
	exit 1
fi

NUM_OUTPUTS=$1
RUN_DIR=$2
OUTPUT_DIR=$3
CONFIG=$4

shift 4

cd ~/magenta
source activate magenta
bazel build --jobs=4 magenta/models/performance_rnn/performance_rnn_generate

# Array of control signals to iterate over
declare -a signatures
signatures[0]="[0.365, 0.575, 2.2600000000000002]"
signatures[1]="[0.05, 0.22000000000000028, 0.6010000000000002]"
signatures[2]="[0.31, -0.11499999999999985, -1.2650000000000001]"
signatures[3]="[-0.34, -0.1789999999999999, 0.13599999999999995]"
signatures[4]="[-0.575, 0.25200000000000033, -0.15999999999999998]"
signatures[5]="[0.3, -0.9590000000000003, -1.873]"
signatures[6]="[0.055, -0.2509999999999998, 0.4039999999999999]"
signatures[7]="[-0.5, 0.575, 2.2600000000000002]"
signatures[8]="[0.0, -1.11, -9.2]"
signatures[9]="[0.75, -0.11599999999999966, -1.2650000000000001]"
signatures[10]="[0.0, -1.6119999999999997, -0.547]"
signatures[11]="[0.0, 0.8590000000000003, 1.0010000000000001]"
signatures[12]="[0.0, -0.1990000000000002, 5.192]"
signatures[13]="[0.25, -0.44600000000000006, -4.417]"
signatures[14]="[0.25, -1.5049999999999997, 1.8079999999999998]"
signatures[15]="[0.0, 3.3549999999999995, -1.0]"
signatures[16]="[0.25, -7.8, 0.7]"

for TEMP in $@
do
	for SIG in "${signatures[@]}"
	do
		./bazel-bin/magenta/models/performance_rnn/performance_rnn_generate \
		--run_dir=$RUN_DIR \
		--output_dir=$OUTPUT_DIR \
		--config=$CONFIG \
		--num_outputs=$NUM_OUTPUTS \
		--temperature=$TEMP \
		--time_place_histogram="${SIG}" \
		--num_steps=6000
	done
done 
