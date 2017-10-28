
from __future__ import absolute_import
from __future__ import print_function

import sys
import tensorflow as tf

from seq2seq import trainer

def main(_):
    with tf.Session() as sess:
        t = trainer.Trainer(sess)
        t.train()

if __name__ == "__main__":
    tf.app.run()
