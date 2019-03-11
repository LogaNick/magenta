"""Generation utility functions."""


class GenerationState():
  """Wrapper class for holding state during generation."""

  def __init__(self):
    # The current time in the generated performance.
    self.time_offset = 0

  def callback(self, encoder_decoder, event_sequences, inputs):
    """Called after each event in a performance is generated. Allows
    control signals to be altered dynamically at generation time.

    Params:
      encoder_decoder: The encoder/decoder object used to generate inputs
        for the model.
      event_sequences: List of events in a performance.
      inputs: List of encoded event inputs.
    """
    # Code to alter control signals.
    pass
