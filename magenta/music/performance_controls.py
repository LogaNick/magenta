# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes for computing performance control signals."""

from __future__ import division

import abc
import copy
import numbers
import ast
import tensorflow as tf
import os

# Used for composer master list
import json
import glob

# internal imports
from magenta.music import constants
from magenta.music import encoder_decoder
from magenta.music.performance_lib import PerformanceEvent

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_NOTE_DENSITY = 15.0
DEFAULT_PITCH_HISTOGRAM = [1.0] * NOTES_PER_OCTAVE
DEFAULT_SIGNATURE_HISTOGRAM = [0.0, 0.0, 0.0]
DEFAULT_TIMEPLACE_VECTOR = [0.0, 0.0, 0.0]
DEFAULT_DATASET_HISTOGRAM = [0.0, 0.0]


# TODO: make this less hacky
if os.path.isfile('/tmp/composer_metadata_master.json'):
  with open('/tmp/composer_metadata_master.json', 'r') as file:
    composer_master_list = json.load(file)

else:
  composer_master_list = []
  composer_list_paths = glob.glob('/tmp/composer_metadata*.json')

  for path in composer_list_paths: 
    with open(path, 'r') as file:
      composer_list = json.load(file)
    for composer in composer_list:
      if composer not in composer_master_list:
        composer_master_list.append(composer)

  with open('/tmp/composer_metadata_master.json', 'w+') as file:
    # Save composer_master_list as JSON
    json.dump(composer_master_list, file)
    
composer_master_list.sort()
COMPOSERS = composer_master_list
DEFAULT_COMPOSER = ''
DEFAULT_COMPOSER_HISTOGRAM = [0.0] * len(COMPOSERS)


class PerformanceControlSignal(object):
  """Control signal used for conditional generation of performances.

  The two main components of the control signal (that must be implemented in
  subclasses) are the `extract` method that extracts the control signal values
  from a Performance object, and the `encoder` class that transforms these
  control signal values into model inputs.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """Name of the control signal."""
    pass

  @abc.abstractproperty
  def description(self):
    """Description of the control signal."""
    pass

  @abc.abstractmethod
  def validate(self, value):
    """Validate a control signal value."""
    pass

  @abc.abstractproperty
  def default_value(self):
    """Default value of the (unencoded) control signal."""
    pass

  @abc.abstractproperty
  def encoder(self):
    """Instantiated encoder object for the control signal."""
    pass

  @abc.abstractmethod
  def extract(self, performance):
    """Extract a sequence of control values from a Performance object.

    Args:
      performance: The Performance object from which to extract control signal
          values.

    Returns:
      A sequence of control signal values the same length as `performance`.
    """
    pass


class NoteDensityPerformanceControlSignal(PerformanceControlSignal):
  """Note density (notes per second) performance control signal."""

  name = 'notes_per_second'
  description = 'Desired number of notes per second.'

  def __init__(self, window_size_seconds, density_bin_ranges):
    """Initialize a NoteDensityPerformanceControlSignal.

    Args:
      window_size_seconds: The size of the window, in seconds, used to compute
          note density (notes per second).
      density_bin_ranges: List of note density (notes per second) bin boundaries
          to use when quantizing. The number of bins will be one larger than the
          list length.
    """
    self._window_size_seconds = window_size_seconds
    self._density_bin_ranges = density_bin_ranges
    self._encoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
        self.NoteDensityOneHotEncoding(density_bin_ranges))

  def validate(self, value):
    return isinstance(value, numbers.Number) and value >= 0.0

  @property
  def default_value(self):
    return DEFAULT_NOTE_DENSITY

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes note density at every event in a performance.

    Args:
      performance: A Performance object for which to compute a note density
          sequence.

    Returns:
      A list of note densities of the same length as `performance`, with each
      entry equal to the note density in the window starting at the
      corresponding performance event time.
    """
    window_size_steps = int(round(
        self._window_size_seconds * performance.steps_per_second))

    prev_event_type = None
    prev_density = 0.0

    density_sequence = []

    for i, event in enumerate(performance):
      if (prev_event_type is not None and
          prev_event_type != PerformanceEvent.TIME_SHIFT):
        # The previous event didn't move us forward in time, so the note density
        # here should be the same.
        density_sequence.append(prev_density)
        prev_event_type = event.event_type
        continue

      j = i
      step_offset = 0
      note_count = 0

      # Count the number of note-on events within the window.
      while step_offset < window_size_steps and j < len(performance):
        if performance[j].event_type == PerformanceEvent.NOTE_ON:
          note_count += 1
        elif performance[j].event_type == PerformanceEvent.TIME_SHIFT:
          step_offset += performance[j].event_value
        j += 1

      # If we're near the end of the performance, part of the window will
      # necessarily be empty; we don't include this part of the window when
      # calculating note density.
      actual_window_size_steps = min(step_offset, window_size_steps)
      if actual_window_size_steps > 0:
        density = (
            note_count * performance.steps_per_second /
            actual_window_size_steps)
      else:
        density = 0.0

      density_sequence.append(density)

      prev_event_type = event.event_type
      prev_density = density

    return density_sequence

  class NoteDensityOneHotEncoding(encoder_decoder.OneHotEncoding):
    """One-hot encoding for performance note density events.

    Encodes by quantizing note density events. When decoding, always decodes to
    the minimum value for each bin. The first bin starts at zero note density.
    """

    def __init__(self, density_bin_ranges):
      """Initialize a NoteDensityOneHotEncoding.

      Args:
        density_bin_ranges: List of note density (notes per second) bin
            boundaries to use when quantizing. The number of bins will be one
            larger than the list length.
      """
      self._density_bin_ranges = density_bin_ranges

    @property
    def num_classes(self):
      return len(self._density_bin_ranges) + 1

    @property
    def default_event(self):
      return 0.0

    def encode_event(self, event):
      for idx, density in enumerate(self._density_bin_ranges):
        if event < density:
          return idx
      return len(self._density_bin_ranges)

    def decode_event(self, index):
      if index == 0:
        return 0.0
      else:
        return self._density_bin_ranges[index - 1]


class ComposerHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Composer class histogram performance control signal."""

  name = 'composer_class_histogram'
  description = "Desired weight for each for each composer"

  def __init__(self, composers):
    """Initializes a ComposerHistogramPerformanceControlSignal.

    Args:
      composers: List of all possible composers for this model
    """
    self._composers = composers
    self._encoder = self.ComposerHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_COMPOSER_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(COMPOSERS) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates composer class histogram at every event in a performance.

    Args:
      performance: A Performance object for which to create a composer class
          histogram sequence.

    Returns:
      A list of composer class histograms the same length as `performance`, 
      where each composer class histogram is the length of the composer 
      master-list of float values. 
      The values sum to one.
    """
    # get list of composers for the given performance
    composer_list_str = ''
    for char in performance.composers:
      composer_list_str += char
    
    # parse string representation of list as a list
    composer_list = ast.literal_eval(composer_list_str)

    # weight each of the composers equally in the histogram
    weight = 1.0 / len(composer_list) 
    default_weight = 0.0

    histogram = []
      
    for composer in COMPOSERS:
      if composer in composer_list:
        histogram.append(weight) # new weights - sum to 1.0
      else:
        histogram.append(default_weight) # currently 0.0 
    
    histogram_sequence = [histogram] * len(performance)
    return histogram_sequence

  class ComposerHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for composer class histogram sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_COMPOSER_HISTOGRAM)

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position]

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class SignatureHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Time signature class histogram performance control signal. Current 
  implementation only distinguishes between even, odd and n/a numerator for
  the time signature."""

  name = 'signature_class_histogram'
  description = "Desired weight for each time signature bin"

  def __init__(self):
    """Initializes a SignatureHistogramPerformanceControlSignal."""
    self._encoder = self.SignatureHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_SIGNATURE_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_SIGNATURE_HISTOGRAM) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates signature class histogram at every event in a performance.

    Args:
      performance: A Performance object for which to create a signature class
          histogram sequence.
      epsilon: small value used to add degree of uncertainty about time signature

    Returns:
      A list of signature class histograms the same length as `performance`, where
      each signature class histogram is the length of the default signature histogram
      of float values. 

      The values sum to one. 
      
      Format: [unknown, 2, 3]
    """

    # get signature for the given performance
    signature_str = ''
    for char in performance.sig_numerator:
      signature_str += char

    # get literal
    signature_numerator = ast.literal_eval(signature_str)

    # get histogram corresponding to time sig numerator
    if signature_numerator is None:
        histogram = [1.00, 0.00, 0.00]
    elif int(signature_numerator) == 2: 
      histogram = [0.05, 0.95, 0.00]
    elif int(signature_numerator) == 3:
      histogram = [0.05, 0.00, 0.95]
    elif int(signature_numerator) == 4:
      histogram = [0.05, 0.95, 0.00]
    elif int(signature_numerator) == 5:
      histogram = [0.70, 0.15, 0.15]
    elif int(signature_numerator) == 6:
      histogram = [0.10, 0.10, 0.80]
    elif int(signature_numerator) == 7:
      histogram = [0.70, 0.20, 0.10]
    elif int(signature_numerator) == 8:
      histogram = [0.05, 0.95, 0.00]
    elif int(signature_numerator) == 9:
      histogram = [0.05, 0.00, 0.95]
    elif int(signature_numerator) == 10:
      histogram = [0.70, 0.15, 0.15]
    elif int(signature_numerator) == 11:
      histogram = [1.00, 0.00, 0.00]
    elif int(signature_numerator) == 12:
      histogram = [0.20, 0.40, 0.40]
    else:
      histogram = [1.00, 0.00, 0.00]
      tf.logging.warning("Time signature numerator: {}".format(signature_numerator))

    histogram_sequence = [histogram] * len(performance)
    
    return histogram_sequence

  class SignatureHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for composer class histogram sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_SIGNATURE_HISTOGRAM)

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position] # TODO: double check this

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class GlobalPositionPerformanceControlSignal(PerformanceControlSignal):
  """Global position performance control signal."""

  name = 'global_position'
  description = 'Desired position vector for performance'

  def __init__(self):
    """Initialize a GlobalPositionPerformanceControlSignal.

    Format: [time from start, time till end]

    Args:
    """
    self._encoder = self.GlobalPositionEncoderDecoder()

  def validate(self, value):
    return isinstance(value, list) and all(isinstance(item, numbers.Number) 
          for item in value)

  @property
  def default_value(self):
    return [0.0, 0.0] # TODO: check this

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes global position at every event in a performance.

    Args:
      performance: A Performance object for which to compute a global
          position at each event.

    Returns:
      A list of vectors of the same length as `performance`, with each
      entry equal to the note density in the window starting at the
      corresponding performance event time.
    """
    delta_time = performance.start_step
    position_sequence = []

    for event in performance:
      if (event.event_type == PerformanceEvent.TIME_SHIFT):
        # The previous event didn't move us forward in time, so the position
        # here should be the same.
        delta_time += event.event_value

      position_sequence.append(delta_time)

    total_time = delta_time

    # Include time till end
    position_sequence = [[t, total_time - t] for t in position_sequence]

    return position_sequence 

  class GlobalPositionEncoderDecoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for global position sequences."""

    @property
    def input_size(self):
      return 2 # time since start, time till end

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position]

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class TimePlacePerformanceControlSignal(PerformanceControlSignal):
  """Year of birth (yob), latitude (lat) and longitude (lon) vector 
  performance control signal. Values are normalized."""

  name = 'time_place_vector'
  description = "Desired normalized vector for year of birth and latitude/longitude"

  def __init__(self):
    """Initializes a TimePlaceHistogramPerformanceControlSignal.

    Args:
      
    """
    self._encoder = self.TimePlaceEncoder()

  @property
  def default_value(self):
    tf.logging.warning("Default time-place vector is being used.")
    return DEFAULT_TIMEPLACE_VECTOR

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_TIMEPLACE_VECTOR) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  @staticmethod
  def normalize(yob, lat, lon):
    """ Normalizes year of birth latitude and longitude. Uses macros.

    Args:
      yob: year of birth
      lat: latitude
      lon: longitude

    Returns:
      Normalized yob, lat, lon
    """

    SCALE_LAT, SHIFT_LAT = 10, 50
    SCALE_LON, SHIFT_LON = 10, 15
    SCALE_YEAR, SHIFT_YEAR = 200, 1800

    yob = (yob - SHIFT_YEAR) / SCALE_YEAR
    lat = (lat - SHIFT_LAT) / SCALE_LAT
    lon = (lon - SHIFT_LON) / SCALE_LON

    return yob, lat, lon


  def extract(self, performance):
    """Creates timeplace histogram at every event in a performance.

    Args:
      performance: A Performance object for which to create a signature class
          histogram sequence.

    Returns:
      A list of timeplace histograms the same length as `performance`, where
      each timeplace histogram is the length of the default timeplace histogram
      of float values. 

      The values are normalized. 
      
      Format: [yob, lat, long]
    """

    # get yob for the given performance
    yob_str = ''
    for char in performance.yob:
      yob_str += char

    # get lat for the given performance
    lat_str = ''
    for char in performance.lat:
      lat_str += char

    # get lon for the given performance
    lon_str = ''
    for char in performance.lon:
      lon_str += char

    # get literal
    try:
      yob, lat, lon = ast.literal_eval(yob_str), ast.literal_eval(lat_str), ast.literal_eval(lon_str)
    except:
      tf.logging.error("AST literal evaluation failed.")
    
    yob, lat, lon = self.normalize(yob, lat, lon)
    vector = [yob, lat, lon]

    vector_sequence = [vector] * len(performance)

    tf.logging.debug("yob={}, lat={}, lon={}".format(yob_str, lat_str, lon_str))
    tf.logging.debug("Length of vector sequence: {}".format(len(vector_sequence)))
    tf.logging.debug("First vector in sequence: {}".format(vector_sequence[0]))
    tf.logging.debug("Middle vector in sequence: {}".format(vector_sequence[len(performance // 2)]))
    tf.logging.debug("Last vector in sequence: {}".format(vector_sequence[-1]))
      
    return vector_sequence

  class TimePlaceEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for sequences of time/place of composition."""

    @property
    def input_size(self):
      return len(DEFAULT_TIMEPLACE_VECTOR)

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position] # TODO: double check this

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class PitchHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Pitch class histogram performance control signal."""

  name = 'pitch_class_histogram'
  description = 'Desired weight for each for each of the 12 pitch classes.'

  def __init__(self, window_size_seconds, prior_count=0.01):
    """Initializes a PitchHistogramPerformanceControlSignal.

    Args:
      window_size_seconds: The size of the window, in seconds, used to compute
          each histogram.
      prior_count: A prior count to smooth the resulting histograms. This value
          will be added to the actual pitch class counts.
    """
    self._window_size_seconds = window_size_seconds
    self._prior_count = prior_count
    self._encoder = self.PitchHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_PITCH_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and len(value) == NOTES_PER_OCTAVE and
            all(isinstance(a, numbers.Number) for a in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes local pitch class histogram at every event in a performance.

    Args:
      performance: A Performance object for which to compute a pitch class
          histogram sequence.

    Returns:
      A list of pitch class histograms the same length as `performance`, where
      each pitch class histogram is a length-12 list of float values summing to
      one.
    """
    window_size_steps = int(round(
        self._window_size_seconds * performance.steps_per_second))

    prev_event_type = None
    prev_histogram = self.default_value

    base_active_pitches = set()
    histogram_sequence = []

    for i, event in enumerate(performance):
      # Maintain the base set of active pitches.
      if event.event_type == PerformanceEvent.NOTE_ON:
        base_active_pitches.add(event.event_value)
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        base_active_pitches.discard(event.event_value)

      if (prev_event_type is not None and
          prev_event_type != PerformanceEvent.TIME_SHIFT):
        # The previous event didn't move us forward in time, so the histogram
        # here should be the same.
        histogram_sequence.append(prev_histogram)
        prev_event_type = event.event_type
        continue

      j = i
      step_offset = 0

      active_pitches = copy.deepcopy(base_active_pitches)
      histogram = [self._prior_count] * NOTES_PER_OCTAVE

      # Count the total duration of each pitch class within the window.
      while step_offset < window_size_steps and j < len(performance):
        if performance[j].event_type == PerformanceEvent.NOTE_ON:
          active_pitches.add(performance[j].event_value)
        elif performance[j].event_type == PerformanceEvent.NOTE_OFF:
          active_pitches.discard(performance[j].event_value)
        elif performance[j].event_type == PerformanceEvent.TIME_SHIFT:
          for pitch in active_pitches:
            histogram[pitch % NOTES_PER_OCTAVE] += (
                performance[j].event_value / performance.steps_per_second)
          step_offset += performance[j].event_value
        j += 1

      histogram_sequence.append(histogram)

      prev_event_type = event.event_type
      prev_histogram = histogram

    return histogram_sequence

  class PitchHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for pitch class histogram sequences."""

    @property
    def input_size(self):
      return NOTES_PER_OCTAVE

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      # Normalize by the total weight.
      total = sum(events[position])
      if total > 0:
        return [count / total for count in events[position]]
      else:
        return [1.0 / NOTES_PER_OCTAVE] * NOTES_PER_OCTAVE

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class DatasetHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Dataset class histogram performance control signal."""
  
  name = 'dataset_class_histogram'
  description = 'Desired weight for each dataset class'

  def __init__(self):
    """Initializes a DatasetHistogramPerformanceControlSignal."""
    self._encoder = self.DatasetHistogramEncoder()
 
  @property
  def default_value(self):
    return DEFAULT_DATASET_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and len(value) == 
           len(DEFAULT_DATASET_HISTOGRAM) and
           all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates dataset histogram at every event in a performance.

    Args:
      performance: A performance object for which to compute a dataset
        class histogram sequence.

    Returns:
      A list of dataset histograms the same length as `performance`, where
      each dataset histogram is the length of the default dataset histogram.
    """
    dataset = performance.dataset
    if dataset == 'yamaha':
      histogram = [1, 0]
    else:
      histogram = [0, 1]
    
    histogram_sequence = [histogram] * len(performance)
    
    return histogram_sequence

  class DatasetHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for dataset class histogram sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_DATASET_HISTOGRAM) 

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position]

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class TempoConditioningFlag(PerformanceControlSignal):
  """Binary flag indicating if a tempo conditioning signal is currently
  being provided; `[1]` if a tempo conditioning signal is present, `[0]`
  otherwise.
  """
  name = 'tempo_flag'
  description = ('Flag indicating if a tempo conditioning signal is '
                 'currently being provided.')

  def __init__(self):
    """Initializes a TempoConditioningFlag."""
    self._encoder = self.TempoConditioningFlagEncoder()
 
  @property
  def default_value(self):
    return [0]

  def validate(self, value):
    return (isinstance(value, list) and 
            len(value) == 1 and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates a tempo conditioning flag for each event in a performance.

    Args:
      performance: performance_lib.Performance object to compute sequence
        of tempo conditioning flags for.

    Returns:
      List of tempo conditioning flags with the same length as `performance`.
    """
    tempo_flag = [1] if performance.tempo_flag else [0]
    return [tempo_flag] * len(performance)

  class TempoConditioningFlagEncoder(
    encoder_decoder.EventSequenceEncoderDecoder):
    """Encoder for TempoConditioningFlag."""

    @property
    def input_size(self):
      # Single bit flag.
      return 1

    @property
    def num_classes(self):
      raise NotImplementedError

    @property
    def default_event_label(self):
      raise NotImplementedError

    def events_to_input(self, events, position):
      return events[position]
    
    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class TempoControlSignal(PerformanceControlSignal):
  """Tempo control signal."""

  name = 'tempo_signal'
  description = 'Tempo conditioning signal.'

  def __init__(self):
    """Initializes a TempoControlSignal."""
    self._encoder = self.TempoControlSignalEncoder()

  @property
  def default_value(self):
    return [0]

  def validate(self, value):
    return (isinstance(value, list) and 
            len(value) == 1 and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes local tempo control signal at each event in a performance.

    Params:
      performance: performance_lib.Performance object for which to compute
        the control signals for.

    Returns:
      A list of control signals the same length as `performance`.
    """
    steps = performance.steps
    histogram_sequence = [[0]] * len(steps)

    if not performance.tempo_flag:
      return histogram_sequence

    for i, event in enumerate(performance):
      time_shift = steps[i]
      indexes = [i for i, val in enumerate(steps) if val == time_shift]

      # Convert time_shift from performance representation to seconds.
      time_shift *= 10
      time_shift /= 1000

      possible_tempos = list(filter(lambda t: t.time <= time_shift, 
        performance.tempos))
      
      #TODO(ncmeade): Compute statistics and pass in.
      scalar_tempo = (possible_tempos[-1].qpm - 7.61) / (283.97 - 7.61)

      for j in indexes:
        histogram_sequence[j] = [scalar_tempo]

    return histogram_sequence


  class TempoControlSignalEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """Encoder for TempoControlSignal."""

    @property
    def input_size(self):
      return 1

    @property
    def default_event_label(self):
      return NotImplementedError

    def events_to_input(self, events, position):
      return events[position]

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


# List of performance control signal classes.
all_performance_control_signals = [
    NoteDensityPerformanceControlSignal,
    PitchHistogramPerformanceControlSignal,
    ComposerHistogramPerformanceControlSignal,
    SignatureHistogramPerformanceControlSignal,
    TimePlacePerformanceControlSignal,
    GlobalPositionPerformanceControlSignal,
    DatasetHistogramPerformanceControlSignal,
    TempoConditioningFlag,
    TempoControlSignal
]
