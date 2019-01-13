#!/bin/bash

# Usage: ./midi_to_example.sh parent_dir_of_midis desired_output_dir
# -----------------------------------------------------------------------------

if (( $# < 3 )); then
	echo "Usage: $0 <num processess> <dir_of_midis> <output_dir> "
	exit 1
elif (( $1 < 1 )); then
	echo "Error: Must specify at least 1 job"
	exit 1
fi

PROCESSES=$1
INPUT_DIRECTORY=$2
OUTPUT_DIRECTORY=$3
TEMP_DIR=~/temp_script_files
TEMP_DIR_IN=$TEMP_DIR/all_inputs
TEMP_DIR_OUT=$TEMP_DIR/all_outputs
TEMP_DIR_OUT_NS=$TEMP_DIR_OUT/all_notesequences
TEMP_DIR_OUT_EX=$TEMP_DIR_OUT/all_sequence_examples

# Setup environment
source activate magenta
cd ~/magenta

# Build executables
bazel build --jobs=${PROCESSES} magenta/scripts:convert_dir_to_note_sequences
bazel build --jobs=${PROCESSES} magenta/models/performance_rnn:performance_rnn_create_dataset

# Make directories
mkdir $TEMP_DIR
mkdir $TEMP_DIR_IN
mkdir $TEMP_DIR_OUT
mkdir $TEMP_DIR_OUT_NS 
mkdir $TEMP_DIR_OUT_EX

declare i=0

# Create n temporary directories
while (( $i < $PROCESSES  )); do
	mkdir $TEMP_DIR_IN/inputs${i}

	(( i++ ))
done

# Disperse hardlinks to all MIDIs evenly between the temp fils
i=0

for FILE in $INPUT_DIRECTORY/*.mid
do
  filename="${FILE%.*}"
	ln $FILE $TEMP_DIR_IN/inputs$(( i % PROCESSES ))/`basename "$FILE"`
  ln ${filename}.json $TEMP_DIR_IN/inputs$(( i % PROCESSES ))/`basename ${filename}.json`
  # filename=$(basename -- "$FILE")
  # extension="${filename##*.}"
  # filename="${FILE%.*}"
  # echo "${filename}.json"
  # echo $(basename ${filename}.json .json)

	(( i++ ))
done


i=0


# Convert MIDIs in each temp file to notesequence TF-records
echo "Creating notesequences . . ."
while (( $i < $PROCESSES )); do
	./dataset_creation_scripts/midi_to_tfrecord.sh $TEMP_DIR_IN/inputs${i} $TEMP_DIR_OUT_NS/notesequences${i}.tfrecord

	(( i++ ))
done

i=0

# Convert notesequences to sequence examples
echo "Creating sequenceexamples . . ."
while (( $i < $PROCESSES )); do

	# Note: this runs in the background
	./dataset_creation_scripts/tfrecord_to_examples.sh $TEMP_DIR_OUT_NS/notesequences${i}.tfrecord $TEMP_DIR_OUT_EX/sequenceexamples${i}&

	(( i++ ))
done

wait

# Concatenate all the TF records into one
echo "Concatenating sequenceexamples . . ."

# Copy the first training and eval tf records to destination dir
cp -r $TEMP_DIR_OUT_EX/sequenceexamples0/ $OUTPUT_DIRECTORY

i=1

while (( $i < $PROCESSES )); do

	cat $TEMP_DIR_OUT_EX/sequenceexamples${i}/training_performances.tfrecord >> $OUTPUT_DIRECTORY/training_performances.tfrecord
	cat $TEMP_DIR_OUT_EX/sequenceexamples${i}/eval_performances.tfrecord >> $OUTPUT_DIRECTORY/eval_performances.tfrecord

	(( i++ ))
done

# Clean up
rm -r $TEMP_DIR
# rm ~/magenta/nohup.out
