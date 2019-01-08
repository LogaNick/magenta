#!/bin/bash

if [[ $# < 3 ]]; then
	echo "Usage: $0 <num_proc> <midi_dir> <example_dir> "
	exit 1
fi

num_proc=$1
input_dir=$2
output_dir=$3
temp_dir=~/temp_script_files
temp_dir_in=$temp_dir/all_inputs
temp_dir_out=$temp_dir/all_outputs
temp_dir_out_ns=$temp_dir_out/all_notesequences
temp_dir_out_ex=$temp_dir_out/all_sequence_examples

cd $HOME/magenta
source activate magenta

bazel build --jobs=${num_proc} magenta/scripts:convert_dir_to_note_sequences
bazel build --jobs=${num_proc} magenta/models/performance_rnn:performance_rnn_create_dataset

mkdir $temp_dir
mkdir $temp_dir_in
mkdir $temp_dir_out
mkdir $temp_dir_out_ns 
mkdir $temp_dir_out_ex

i=0
while (( $i < $num_proc  )); do
	mkdir $temp_dir_in/inputs${i}
	(( i++ ))
done

# Disperse MIDIs via hardlinks.
i=0
for midi_file in $input_dir/*.mid; do
  filename="${midi_file%.*}"
  # Link MIDI to temp.
	ln $midi_file $temp_dir_in/inputs$(( i % num_proc ))/`basename "$midi_file"`
  # Link JSON metadata to temp.
  # ln ${filename}.json $temp_dir_in/inputs$(( i % num_proc ))/`basename ${filename}.json`
	(( i++ ))
done

# Convert MIDIs in each temp to notesequences.
i=0
echo "Creating notesequences . . ."
while (( $i < $num_proc )); do
  ./bazel-bin/magenta/scripts/convert_dir_to_note_sequences \
    --input_dir=$temp_dir_in/inputs${i} \
    --output_file=$temp_dir_out_ns/notesequences${i}.tfrecord \
    --recursive
	(( i++ ))
done

# Convert notesequences to sequence examples.
i=0
echo "Creating sequenceexamples . . ."
while (( $i < $num_proc )); do
  ./bazel-bin/magenta/models/performance_rnn/performance_rnn_create_dataset \
    --config=tempo_conditioned_performance_with_dynamics \
    --input=$temp_dir_out_ns/notesequences${i}.tfrecord \
    --output_dir=$temp_dir_out_ex/sequenceexamples${i} \
    --eval_ratio=0.10 &
	(( i++ ))
done
wait

echo "Concatenating sequenceexamples . . ."

touch $output_dir/training_performances.tfrecord \
      $output_dir/eval_performances.tfrecord

i=0
while (( $i < $num_proc )); do
  cat $temp_dir_out_ex/sequenceexamples${i}/training_performances.tfrecord \
    >> $output_dir/training_performances.tfrecord
  cat $temp_dir_out_ex/sequenceexamples${i}/eval_performances.tfrecord \
    >> $output_dir/eval_performances.tfrecord
	(( i++ ))
done

# Clean up.
rm -r $temp_dir
