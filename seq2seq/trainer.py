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
from seq2seq import model as m
from util import attr_reader

with open('config.json') as f:
    FLAGS = attr_reader.AttrReader(json.loads(f.read()))

class Trainer():
    def __init__(self, sess):
        self.sess = sess
        self._buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

    def read_data(self, source_path, target_path, max_size=None):
        """Read data from source and target files and put into buckets.

        Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

        Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
        """
        data_set = [[] for _ in self._buckets]
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                      print("  reading data line %d" % counter)
                      sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(self._buckets):
                      if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                    source, target = source_file.readline(), target_file.readline()
                    return data_set

    def train(self):
        """Train a seq->seq model using two sets of training."""
        from_train = None
        to_train = None
        from_dev = None
        to_dev = None
        from_train_data = FLAGS.from_train_data
        to_train_data = FLAGS.to_train_data
        from_dev_data = from_train_data
        to_dev_data = to_train_data
        from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
            FLAGS.data_dir,
            from_train_data,
            to_train_data,
            from_dev_data,
            to_dev_data,
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size)
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = m.create_model(self.sess, self._buckets, False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)."
               % FLAGS.max_train_data_size)
        dev_set = self.read_data(from_dev, to_dev)
        train_set = self.read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(self._buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)
          _, step_loss, _ = model.step(self.sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, False)
          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                             step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              self.sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(self.sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(self._buckets)):
              if len(dev_set[bucket_id]) == 0:
                print("  eval: empty bucket %d" % (bucket_id))
                continue
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  dev_set, bucket_id)
              _, eval_loss, _ = model.step(self.sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
              eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                  "inf")
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()
