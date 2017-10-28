from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from seq2seq import data_utils
from seq2seq import model
from util import attr_reader

with open('config.json') as f:
    FLAGS = attr_reader.AttrReader(json.loads(f.read()))

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.

class Decoder():
    def __init__(self, sess):
        self.sess = sess
        self._buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        self.model = model.create_model(self.sess, self._buckets, True)
        self.model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        from_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.from_vocab_size)
        to_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.to" % FLAGS.to_vocab_size)
        self.from_vocab, self._ = data_utils.initialize_vocabulary(from_vocab_path)
        self._, self.rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)

    def decode(self, sentence):
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), self.from_vocab)
      # Which bucket does it belong to?
      bucket_id = len(self._buckets) - 1
      for i, bucket in enumerate(self._buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      self._,self. _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      return " ".join([tf.compat.as_str(self.rev_to_vocab[output]) for output in outputs])
