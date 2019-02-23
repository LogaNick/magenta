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

# internal imports
from magenta.music import constants
from magenta.music import encoder_decoder
from magenta.music.performance_lib import PerformanceEvent

NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_NOTE_DENSITY = 15.0
DEFAULT_PITCH_HISTOGRAM = [1.0] * NOTES_PER_OCTAVE
DEFAULT_SIGNATURE_HISTOGRAM = [0.0, 0.0, 0.0]
DEFAULT_TIMEPLACE_VECTOR = [0.0, 0.0, 0.0]
DEFAULT_DATASET_SIGNAL = [0.0, 0.0]
COMPOSERS = constants.COMPOSER_SET
DEFAULT_COMPOSER_HISTOGRAM = [0.0] * len (COMPOSERS)
DEFAULT_VELOCITY_HISTOGRAM = [0.0, 0.0, 0.0, 0.0, 0.0]
COMPOSER_CLUSTERS = constants.COMPOSER_CLUSTERS
DEFAULT_COMPOSER_CLUSTER = [0.0] * (len(COMPOSER_CLUSTERS) + 1)
MAJOR_MINOR_VECTOR = ['major', 'minor']
TEMPO_KEYWORDS = ['allegro', 'allegretto', 'andante', 'adagio', 'presto']
RANKED_TEMPO_KEYWORDS = ['adagio', 'andante', 'allegretto', 'allegro', 'presto'] 
# the last two componenets of DEFAULT_TEMPO_WORD_VECTOR represent 'is mixed tempo'
# and 'is unknown' respectively
assert len(TEMPO_KEYWORDS) > 0
DEFAULT_TEMPO_WORD_VECTOR = [1 / len(TEMPO_KEYWORDS)] * len(TEMPO_KEYWORDS) 
DEFAULT_TEMPO_BOOST_VECTOR = [0] * len(TEMPO_KEYWORDS)
FORM_KEYWORDS = ['prelude',	'fugue', 'waltz',	'etude',	'variations',	'scherzo',	'impromptu',	'ballade',	'toccata',	'polonaise',	'nocturne',	'hungarian',	'dance',	'espagnol',	'intermezzo',	'mazurka']
assert len(FORM_KEYWORDS) > 0
DEFAULT_FORM_WORD_VECTOR = [1 / len(FORM_KEYWORDS)] * len(FORM_KEYWORDS)


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

class MajorMinorPerformanceControlSignal(PerformanceControlSignal):
  """Major or minor performance control signal."""

  name = 'major_or_minor'
  description = 'Desired key: major or minor'

  def __init__(self, vector=MAJOR_MINOR_VECTOR):
    """Initialize a MajorMinorPerformanceControlSignal.

    Args: 
      vector - list of possible values
    """
    self.vector = vector
    self._encoder = self.MajorMinorHistogramEncoder()

  def validate(self, value):
    return value in self.vector

  @property
  def default_value(self):
    return None

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Extracts whether performance is major/minor and applies this at every event.

    Args:
      performance: A Performance object for which to annotate.

    Returns:
      A list with elements 'major', 'minor' or None depending on whether or not the 
      performance is major, minor, or the key is unknown respectively.
    """  

    # TODO: fix this hack
    key_sig = ''
    for c in performance.key_signature:
      key_sig += c

    if 'major' in key_sig:
      return [[1.0, 0.0]] * len(performance)
    elif 'minor' in key_sig:
      return [[0.0, 1.0]] * len(performance)
    else:
      return [[0.5,0.5]] * len(performance)

  class MajorMinorHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for composer class histogram sequences."""

    @property
    def input_size(self):
      return len(MAJOR_MINOR_VECTOR)

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
    #TODO(NicholasBarreyre): see if there is a function for this
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

class TempoWordPerformanceControlSignal(PerformanceControlSignal):
  """Tempo keyword performance control signal."""

  name = 'tempo_keyword_vector'
  description = "Desired tempo"

  def __init__(self):
    """Initializes a TempoWordPerformanceControlSignal.

    Args:
    """
    self._encoder = self.TempoWordHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_TEMPO_WORD_VECTOR

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_TEMPO_WORD_VECTOR) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates tempo keyword vector at every event in a performance.

    Args:
      performance: A Performance object for which to create a tempo vector
          sequence.

    Returns:
      A list of tempo keyword vectors the same length as `performance`.
    """
    if not performance.keywords:
      return [DEFAULT_TEMPO_WORD_VECTOR] * len(performance)

    #TODO(NicholasBarreyre): see if there is a function for this
    keyword_list_str = ''
    for char in performance.keywords:
      keyword_list_str += char
    
    # parse string representation of list as a list
    keywords = ast.literal_eval(keyword_list_str)

    # Extract keyword associated with tempo
    tempo_indicators = []
    for keyword in keywords:
      if keyword in TEMPO_KEYWORDS:
        tempo_indicators.append(keyword)

    
    if len(tempo_indicators) == 0:
      vector = DEFAULT_TEMPO_WORD_VECTOR
    else: 
      vector = []

      # weight each of the tempo indicators equally in the vector
      weight = 1.0 / len(tempo_indicators)
      default_weight = 0.0

      for tempo_indicator in TEMPO_KEYWORDS:
        if tempo_indicator in tempo_indicators:
          vector.append(weight)
        else:
          vector.append(default_weight)
    
    vector_sequence = [vector] * len(performance)
    return vector_sequence

  class TempoWordHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for tempo word vector sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_TEMPO_WORD_VECTOR)

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

class TempoBoostPerformanceControlSignal(PerformanceControlSignal):
  """Tempo boost performance control signal."""

  name = 'tempo_boost_vector'
  description = "Desired tempo"

  def __init__(self):
    """Initializes a TempoBoostPerformanceControlSignal.

    Args:
    """
    self._encoder = self.TempoBoostHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_TEMPO_BOOST_VECTOR

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_TEMPO_WORD_VECTOR) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates tempo keyword vector at every event in a performance.

    Args:
      performance: A Performance object for which to create a tempo vector
          sequence.

    Returns:
      A list of tempo keyword vectors the same length as `performance`.
    """
    if not performance.keywords:
      return [DEFAULT_TEMPO_BOOST_VECTOR] * len(performance)

    #TODO(NicholasBarreyre): see if there is a function for this
    keyword_list_str = ''
    for char in performance.keywords:
      keyword_list_str += char
    
    # parse string representation of list as a list
    keywords = ast.literal_eval(keyword_list_str)

    # Extract keyword associated with tempo
    tempo_indicators = []
    for keyword in keywords:
      if keyword in TEMPO_KEYWORDS:
        tempo_indicators.append(keyword)

    
    if len(tempo_indicators) == 0:
      vector = DEFAULT_TEMPO_BOOST_VECTOR
    else: 
      vector = []

      # weight each of the tempo indicators equally in the vector
      weight = 1.0 / len(tempo_indicators)
      default_weight = 0.0

      for tempo_indicator in TEMPO_KEYWORDS:
        if tempo_indicator in tempo_indicators:
          vector.append(weight)
        else:
          vector.append(default_weight)
    
    vector_sequence = [vector] * len(performance)
    return vector_sequence

  class TempoBoostHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for tempo word vector sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_TEMPO_BOOST_VECTOR)

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

class TempoRankHotPerformanceControlSignal(PerformanceControlSignal):
  """Tempo rank-hot performance control signal."""

  name = 'tempo_rank_hot'
  description = "Desired tempo rank-hot"

  def __init__(self):
    """Initializes a TempoBoostPerformanceControlSignal.

    Args:
    """
    self._encoder = self.TempoRankHotEncoder()

  @property
  def default_value(self):
    return DEFAULT_TEMPO_BOOST_VECTOR

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_TEMPO_WORD_VECTOR) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates tempo rank hot vector at every event in a performance.

    Args:
      performance: A Performance object for which to create a tempo vector
          sequence.

    Returns:
      A list of tempo rank hot vectors the same length as `performance`.
    """
    if not performance.keywords:
      return [DEFAULT_TEMPO_BOOST_VECTOR] * len(performance)

    #TODO(NicholasBarreyre): see if there is a function for this
    keyword_list_str = ''
    for char in performance.keywords:
      keyword_list_str += char
    
    # parse string representation of list as a list
    keywords = ast.literal_eval(keyword_list_str)

    # Extract keyword associated with tempo
    tempo_indicators = []
    for keyword in keywords:
      if keyword in TEMPO_KEYWORDS:
        tempo_indicators.append(keyword)

    
    vector = DEFAULT_TEMPO_BOOST_VECTOR

    if len(tempo_indicators) != 0: 

      # weight each of the tempo indicators equally in the vector
      weight = 1.0 / len(tempo_indicators)

      # index of the halfway point (equivalent to the ceiling)
      half_way = len(RANKED_TEMPO_KEYWORDS) // 2

      for tempo in tempo_indicators:
        tempo_index = RANKED_TEMPO_KEYWORDS.index(tempo)

        i = half_way
        if tempo_index < half_way:      
          while i >= tempo_index:
            vector[i] += weight
            i -= 1
        else:
          while i <= tempo_index:
            vector[i] += weight
            i += 1
    
    vector_sequence = [vector] * len(performance)
    return vector_sequence

  class TempoRankHotEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for tempo rank-hot vector sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_TEMPO_BOOST_VECTOR)

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

class FormWordPerformanceControlSignal(PerformanceControlSignal):
  """Form keyword performance control signal."""

  name = 'form_keyword_vector'
  description = "Desired form"

  def __init__(self):
    """Initializes a FormWordPerformanceControlSignal.

    Args:
    """
    self._encoder = self.FormWordHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_FORM_WORD_VECTOR

  def validate(self, value):
    return (isinstance(value, list) and len(value) == len(DEFAULT_FORM_WORD_VECTOR) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates form keyword vector at every event in a performance.

    Args:
      performance: A Performance object for which to create a form vector
          sequence.

    Returns:
      A list of form keyword vectors the same length as `performance`.
    """
    if not performance.keywords:
      return [DEFAULT_FORM_WORD_VECTOR] * len(performance)

    #TODO(NicholasBarreyre): see if there is a function for this
    keyword_list_str = ''
    for char in performance.keywords:
      keyword_list_str += char

    # We need composers in order to ignore Bach and Handel for forms other than fugue
    composer_list_str = ''
    for char in performance.composers:
      composer_list_str += char
    
    # parse string representation of list as a list
    keywords = ast.literal_eval(keyword_list_str)
    composers = ast.literal_eval(composer_list_str)

    contains_bach_or_handel = 'Bach' in composers or 'Handel' in composers

    # Extract keyword associated with form
    form_indicators = []
    for keyword in keywords:
      if keyword in FORM_KEYWORDS:
        form_indicators.append(keyword)

    if contains_bach_or_handel and 'fugue' not in form_indicators:
      return [DEFAULT_FORM_WORD_VECTOR] * len(performance)

    if len(form_indicators) == 0:
      vector = DEFAULT_FORM_WORD_VECTOR
    else: 
      vector = []

      # weight each of the form indicators equally in the vector
      weight = 1.0 / len(form_indicators)
      default_weight = 0.0

      for form_indicator in FORM_KEYWORDS:
        if form_indicator in form_indicators:
          vector.append(weight)
        else:
          vector.append(default_weight)
    
    vector_sequence = [vector] * len(performance)
    return vector_sequence

  class FormWordHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for form word vector sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_FORM_WORD_VECTOR)

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


class ComposerClusterPerformanceControlSignal(PerformanceControlSignal):
  """Composer cluster histogram performance control signal."""

  name = 'composer_cluster'
  description = "Desired weight for each for each composer cluster"

  def __init__(self, composers):
    """Initializes a ComposerClusterPerformanceControlSignal.

    Args:
      composers: List of all possible composers for this model
    """
    self._composers = composers
    self._encoder = self.ComposerClusterEncoder()

  @property
  def default_value(self):
    return DEFAULT_COMPOSER_CLUSTER

  def validate(self, value):
    return (isinstance(value, list) and len(value) == (len(COMPOSER_CLUSTERS) + 1) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Creates composer cluster histogram at every event in a performance.

    Args:
      performance: A Performance object for which to create a composer cluster
          histogram sequence.

    Returns:
      A list of composer cluster histograms the same length as `performance`, 
      where each composer cluster histogram is the length of the COMPOSER_CLUSTERS 
      of float values. The values sum to one.
    """
    # get list of composers for the given performance
    #TODO(NicholasBarreyre): see if there is a function for this
    composer_list_str = ''
    for char in performance.composers:
      composer_list_str += char
    
    # parse string representation of list as a list
    composer_list = ast.literal_eval(composer_list_str)

    # weight each of the composers equally in the histogram
    weight = 1.0 / len(composer_list) 
    default_weight = 0.0

    histogram = [default_weight] * (len(COMPOSER_CLUSTERS) + 1)

    for composer in composer_list:
      for i, cluster in enumerate(COMPOSER_CLUSTERS):
        if composer in cluster:
          histogram[i] += weight
          break
      else:
        histogram[-1] += weight

    histogram_sequence = [histogram] * len(performance)
    return histogram_sequence

  class ComposerClusterEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for composer class histogram sequences."""

    @property
    def input_size(self):
      return len(COMPOSER_CLUSTERS) + 1

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
    # TODO (NicholasBarreyre): put this in constants file
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

    # TODO(NicholasBarreyre): Put these in constants file
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


class DatasetControlSignal(PerformanceControlSignal):
  """Dataset control signal."""

  name = 'dataset_signal'
  description = 'Dataset conditioning signal.'

  def __init__(self):
    """Initializes a DatasetControlSignal."""
    self._encoder = self.DatasetControlSignalEncoder()

  @property
  def default_value(self):
    return DEFAULT_DATASET_SIGNAL

  def validate(self, value):
    return (isinstance(value, list) and
            len(value) == len(DEFAULT_DATASET_SIGNAL) and
            all(isinstance(val, numbers.Number) for val in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes the local dataset control signal at each event in a
    performance.

    Params:
      performance: performance_lib.Performance object for which to compute
        the control signal for.

    Returns:
      A list of control signals the same length as `performance`.
    """
    if performance.metadata.dataset == 'yamaha':
      signal = [1, 0]
    else:
      signal = [0, 1]

    signals = [signal] * len(performance)

    return signals

  class DatasetControlSignalEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """Encoder for DatasetControlSignal."""

    @property
    def input_size(self):
      return len(DEFAULT_DATASET_SIGNAL)

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
    signals = [[0]] * len(performance.steps)

    for i, event in enumerate(performance):
      time_shift = performance.steps[i]
      indexes = [i for i, val in enumerate(performance.steps) 
        if val == time_shift]

      # Convert the time_shift from performance representation to seconds.
      time_shift *= 10
      time_shift /= 1000

      possible_tempos = list(filter(lambda t: t.time <= time_shift,
        performance.tempos))

      #TODO(ncmeade): Compute statistics and pass in. Currently hard coded
      # with values for Piano-midi dataset.
      scalar_tempo = (possible_tempos[-1].qpm - 7.61) / (283.97 - 7.61)

      for j in indexes:
        signals[j] = [scalar_tempo]

      return signals

  class TempoControlSignalEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """Encoder for TempoControl signal."""

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


class VelocityHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Velocity class histogram performance control signal."""

  name = 'velocity_class_histogram'
  description = 'Desired weight for each of the 3 velocity classes.'

  def __init__(self, window_size_seconds, prior_count=0.01):
    """Initializes a VelocityHistogramPerformanceControlSignal.

    Params:
      window_size_seconds: The size of the window, in seconds, used to compute
          each histogram.
      prior_count: A prior count to smooth the resulting histograms. This
          will be added to the actual velocity class counts.
    """
    self._window_size_seconds = window_size_seconds
    self._prior_count = prior_count
    self._encoder = self.VelocityHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_VELOCITY_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and 
            len(value) == len(DEFAULT_VELOCITY_HISTOGRAM) and 
            all(isinstance(a, numbers.Number) for a in value))
  
  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes a local velocity class histogram at every event in a 
    performance.

    Params:
      performance: A Performance object for which to compute a velocity class
          histogram sequence.

    Returns:
      A list of velocity class histograms the same length as `performance`, 
      where each velocity class histogram is a length 3 list of float values
      summing to one.
    """
    window_size_steps = int(round(
        self._window_size_seconds * performance.steps_per_second))

    initial_velocity = 15  # Median of velocity distribution.
    prev_event_type = None
    prev_histogram = self.default_value

    base_active_pitches = set()
    histogram_sequence = []

    for i, event in enumerate(performance):
      # Initialize to last known velocity.
      prev_velocity = initial_velocity

      # Maintain the base set of active pitches.
      if event.event_type == PerformanceEvent.NOTE_ON:
        base_active_pitches.add((event.event_value, prev_velocity))
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        base_active_pitches = {t for t in base_active_pitches 
            if t[0] != event.event_value}
      elif event.event_type == PerformanceEvent.VELOCITY:
        prev_velocity = event.event_value

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
      histogram = [self._prior_count] * len(DEFAULT_VELOCITY_HISTOGRAM)

      # Count the total duration of each velocity class within the window.
      while step_offset < window_size_steps and j < len(performance):
        if performance[j].event_type == PerformanceEvent.NOTE_ON:
          active_pitches.add((performance[j].event_value, prev_velocity))
        elif performance[j].event_type == PerformanceEvent.NOTE_OFF:
          active_pitches = {t for t in active_pitches 
              if t[0] != event.event_value}
        elif performance[j].event_type == PerformanceEvent.VELOCITY:
          prev_velocity = performance[j].event_value
        elif performance[j].event_type == PerformanceEvent.TIME_SHIFT:
          for pitch in active_pitches:
            # Bin the NOTE_ON.
            # TODO(ncmeade): Extract to function.
            if pitch[1] < 15:
              histogram[0] += (
                  performance[j].event_value / performance.steps_per_second)

            elif pitch[1] >= 15 and pitch[1] <= 19:
              histogram[1] += (
                  performance[j].event_value / performance.steps_per_second)
            else:
              histogram[2] += (
                  performance[j].event_value / performance.steps_per_second)
          step_offset += performance[j].event_value
        j += 1

      histogram_sequence.append(histogram)

      prev_event_type = event.event_type
      prev_histogram = histogram

      # Keep track of the latest velocity while stepping through events.
      if performance[i].event_type == PerformanceEvent.VELOCITY:
        initial_velocity = performance[i].event_value

    return histogram_sequence


  class VelocityHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for velocity class histogram sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_VELOCITY_HISTOGRAM)

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
        return ([1.0 / len(DEFAULT_VELOCITY_HISTOGRAM)] * 
            len(DEFAULT_VELOCITY_HISTOGRAM))

    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


class RelativePositionControlSignal(PerformanceControlSignal):
  """Relative position performance control signal."""

  name = 'relative_position'
  description = 'Relative position in performance.'

  def __init__(self):
    """Initializes a RelativePositionControlSignal."""
    self._encoder = self.RelativePositionEncoderDecoder()

  def validate(self, value):
    return (isinstance(value, list) and
            all(isinstance(v, numbers.Number) for v in value))

  @property
  def default_value(self):
    return [0.0]
  
  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes the local relative position signal at each event in a
    performance.

    Params:
      performance: A Performance object for which to compute the relative
        position at each event.

    Returns:
      List of control signals for each event in a performance.
    """
    # Time offset in seconds.
    time_offset = 0
    position_sequence = []

    # If this performance is an 'ending', append silence.
    if performance.subsequence_info.end_time_offset == 0:
      performance._append_steps(1500)

    for event in performance:
      if event.event_type == PerformanceEvent.TIME_SHIFT:
        # Convert to seconds.
        time_offset += event.event_value / 100

      position = ((performance.subsequence_info.start_time_offset + 
                  time_offset) / 
                  performance.subsequence_info.source_total_time)

      position_sequence.append([position])

    return position_sequence

  class RelativePositionEncoderDecoder(
    encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for the relative position control signal."""

    @property
    def input_size(self):
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

class CenturyControlSignal(PerformanceControlSignal):
  "Year of birth of composer control signal."""

  name = 'century_signal'
  description = 'Century control signal.'

  def __init__(self):
    self._encoder = self.CenturyControlSignalEncoder()

  @property
  def default_value(self):
    return [0, 0, 0, 0]

  @property
  def validate(self, value):
    return (instance(value, list) and
            len(value) == 4 and
            all(isinstance(val, numbers.Number) for val in value))
  
  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes the local control signal at each event in a performance.

    Params:
      performance: A Performance object for which to compute the relative
        position at each event.

    Returns:
      List of control signals for each event in a performance.
    """
    if performance.yob == "None":
      signal = [0, 0, 0, 0]
    elif int(float(performance.yob)) >= 1900:
      signal = [0, 0, 0, 1]
    elif int(float(performance.yob)) >= 1800:
      signal = [0, 0, 1, 0]
    elif int(float(performance.yob)) >= 1700:
      signal = [0, 1, 0, 0]
    elif int(float(performance.yob)) >= 1600:
      signal = [1, 0, 0, 0]

    return [signal] * len(performance)
 
  class CenturyControlSignalEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for the century control signal."""
    
    @property
    def input_size(self):
      return 4

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


class StaticVelocityHistogramPerformanceControlSignal(PerformanceControlSignal):
  """Static velocity class histogram performance control signal."""

  name = 'static_velocity_class_histogram'
  description = 'Desired weight for each of the 3 velocity classes.'

  def __init__(self, bin_boundaries, prior_count=0.01):
    """Initializes a StaticVelocityHistogramPerformanceControlSignal.

    Params:
      prior_count: A prior count to smooth the resulting histograms. This
          will be added to the actual velocity class counts.
      bin_boundaries: The bin boundaries to use when computing the histograms.
    """
    self._prior_count = prior_count
    self._bin_boundaries = bin_boundaries
    self._encoder = self.StaticVelocityHistogramEncoder()

  @property
  def default_value(self):
    return DEFAULT_VELOCITY_HISTOGRAM

  def validate(self, value):
    return (isinstance(value, list) and 
            len(value) == len(DEFAULT_VELOCITY_HISTOGRAM) and 
            all(isinstance(a, numbers.Number) for a in value))

  @property
  def encoder(self):
    return self._encoder

  def extract(self, performance):
    """Computes the velocity class histogram for the performance.

    Params:
      performance: A Performance object for which to compute a velocity class
          histogram sequence.

    Returns:
      A list of velocity class histograms the same length as `performance`,
      where each velocity class histogram is a length 3 list of float values
      summing to one.
    """
    # Initialize velocity to the median of the distribution.
    prev_velocity = 15
    histogram = [self._prior_count] * len(DEFAULT_VELOCITY_HISTOGRAM)

    for i, event in enumerate(performance):
      if event.event_type == PerformanceEvent.VELOCITY:
        prev_velocity = event.event_value
      elif event.event_type == PerformanceEvent.NOTE_ON:
        histogram[self._to_histogram_bin(prev_velocity)] += 1
    
    histogram_sequence = [histogram] * len(performance)

    return histogram_sequence 

  def _to_histogram_bin(self, velocity):
    """Maps the velocity to a histogram bin.

    Params:
      velocity: The velocity to map to a histogram bin.

    Returns:
      The integer index of the bin to which the velocity is mapped.
    """
    for i, (lower_bound, upper_bound) in enumerate(self._bin_boundaries):
      if velocity >= lower_bound and velocity <= upper_bound:
        return i

  class StaticVelocityHistogramEncoder(
      encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for static velocity class histogram sequences."""

    @property
    def input_size(self):
      return len(DEFAULT_VELOCITY_HISTOGRAM)

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
        return ([1.0 / len(DEFAULT_VELOCITY_HISTOGRAM)] *
            len(DEFAULT_VELOCITY_HISTOGRAM))
   
    def events_to_label(self, events, position):
      raise NotImplementedError

    def class_index_to_event(self, class_index, events):
      raise NotImplementedError


# List of performance control signal classes.
all_performance_control_signals = [
    NoteDensityPerformanceControlSignal,
    PitchHistogramPerformanceControlSignal,
    ComposerHistogramPerformanceControlSignal,
    ComposerClusterPerformanceControlSignal,
    SignatureHistogramPerformanceControlSignal,
    TimePlacePerformanceControlSignal,
    TempoControlSignal,
    DatasetControlSignal,
    RelativePositionControlSignal,
    MajorMinorPerformanceControlSignal,
    CenturyControlSignal,
    VelocityHistogramPerformanceControlSignal,
    StaticVelocityHistogramPerformanceControlSignal,
    TempoWordPerformanceControlSignal,
    FormWordPerformanceControlSignal,
    TempoBoostPerformanceControlSignal,
    TempoRankHotPerformanceControlSignal
]
