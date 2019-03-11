import os
import glob
import json
import argparse

def get_composer_set(json_paths):
  """Create a list of disctinct composers from JSON files.

  Alphabetically sorts it.

  Args:
    json_paths: list of file paths to all JSONS

  Returns:
    A list of distinct composers
  """
  composer_set = set()

  for file in json_paths:
    with open(file, 'r') as handle:
      parsed_dict = json.load(handle)
      composers = parsed_dict['composers']

    for composer in composers:
      composer_set.add(composer)


  return sorted(composer_set)

def get_composer_vector(composer_list, composer_name):
  """Create a one-hot vector for the given composer_name

  Args:
    composer_list: list of distinct composers, sorted alphabetically
    composer_name: string; name of composer in composer set

  Returns: a list; a one-hot with only composer_name "on"
  """
  result_vec = [0.0] * len(composer_list)

  for i, composer in enumerate(composer_list):
    if composer == composer_name:
      result_vec[i] = 1.0
  
  return result_vec

def main():

  # get command line args
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, help='Absolute path to input directory.')
  parser.add_argument('--composer', type=str, help='Composer name to map to vector.')
  args = parser.parse_args()

  # get list of all JSON and MIDI files in input directory
  path = os.path.join(os.path.expanduser(args.input_dir), '*.json')
  json_paths = glob.glob(path)
  composer_set = get_composer_set(json_paths)

  if args.composer is not None:
    composer_vec = get_composer_vector(composer_set, args.composer)
    print(composer_vec)
  else:
    print(composer_set)


if __name__ == '__main__':
    main()