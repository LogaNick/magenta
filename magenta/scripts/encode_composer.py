"""This script can be used to generate the command line arguments to use
multi_composer_conditioned_performance_with_dynamics."""

import argparse
import glob
import json

parser = argparse.ArgumentParser(description='Process fields to generate vector string')
parser.add_argument('--master_file', default='/tmp/composer_metadata_master.json', 
        help='file path of the composer master list')
parser.add_argument('--default_val', default=0.0, help='the default value to use in the vector',
        type=float)
parser.add_argument('--specified_val', default=1.0, help='the value to use for the specified composers',
        type=float)
parser.add_argument('composer', help='list of composers to use specified value for')
args = parser.parse_args()


# get master list
with open(args.master_file, 'r') as file:
    composer_master_list = json.load(file)
composer_master_list.sort()

# make vector
output = [args.specified_val if composer==args.composer else args.default_val
            for composer in composer_master_list]

# convert to string and preprocess
output = str(output).replace(' ', '')

print(output)