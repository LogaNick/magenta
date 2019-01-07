#!/bin/bash

PROCESSES=$1
OUTPUT_DIRECTORY=$2
TEMP_DIR=~/temp_script_files
TEMP_DIR_IN=$TEMP_DIR/all_inputs
TEMP_DIR_OUT=$TEMP_DIR/all_outputs
TEMP_DIR_OUT_NS=$TEMP_DIR_OUT/all_notesequences
TEMP_DIR_OUT_EX=$TEMP_DIR_OUT/all_sequence_examples

# Copy the first training and eval tf records to destination dir
cp $TEMP_DIR_OUT_EX/sequenceexamples0/* $OUTPUT_DIRECTORY/sequenceexamples

i=1

while (( $i < $PROCESSES )); do

        cat $TEMP_DIR_OUT_EX/sequenceexamples${i}/training_performances.tfrecord >> $OUTPUT_DIRECTORY/sequenceexamples/training_performances.tfrecord
        cat $TEMP_DIR_OUT_EX/sequenceexamples${i}/eval_performances.tfrecord >> $OUTPUT_DIRECTORY/sequenceexamples/eval_performances.tfrecord

        (( i++ ))
done

