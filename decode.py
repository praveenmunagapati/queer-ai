
from __future__ import absolute_import
from __future__ import print_function

import sys
import tensorflow as tf

from seq2seq import decoder

# Decode from standard input.
sys.stdout.write("> ")
sys.stdout.flush()

def main(_):
    with tf.Session() as sess:
        d = decoder.Decoder(sess)
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
          response = d.decode(sentence)
          print(response)
          print("> ", end="")
          sys.stdout.flush()
          sentence = sys.stdin.readline()

if __name__ == "__main__":
    tf.app.run()
