import os
import csv
import re

INPUT_FILE = "data/literotica.csv"
TRAIN_FILE = "data/train.txt"
TEST_FILE = "data/test.txt"
VALID_FILE = "data/valid.txt"

VALID_FRAC = .01
TEST_FRAC = .01
VOCAB_SIZE = 10000

chars = ""

with open(INPUT_FILE, 'r+') as data_file:
    for line in csv.DictReader(data_file):
        chars += line['story']

def cleanup(chars):
    output = chars.lower()
    output = re.sub("([^.!a-z0-9' ])", "", output)
    output = re.sub("([\.|!]) ?", "\n", output)
    output = re.sub("\n+", " \n ", output)
    output = re.sub(" +", " ", output)
    return output


def limit_vocab(chars, vocab):
    def limit_word(word):
        if word in vocab:
            return word
        else:
            return "<unk>"
    words = chars.split(" ")
    return " ".join(map(limit_word, words))


# remove rare words
word_count = {}
chars = cleanup(chars)
all_words = chars.split(" ")
for w in all_words:
    if w in word_count:
        word_count[w] += 1
    else:
        word_count[w] = 1
vocab = sorted(word_count.iteritems(), key=lambda (k,v): (v,k))
vocab.reverse()
vocab = map(lambda (i): i[0], vocab[:VOCAB_SIZE])

valid_end = int(len(chars) * VALID_FRAC)
test_end = valid_end + int(len(chars) * TEST_FRAC)

valid = limit_vocab(chars[:valid_end], vocab)
test = limit_vocab(chars[valid_end:test_end], vocab)
train = limit_vocab(chars[test_end:], vocab)

with open(TRAIN_FILE, 'w+') as data:
    data.write(train)
with open(TEST_FILE, 'w+') as data:
    data.write(test)
with open(VALID_FILE, 'w+') as data:
    data.write(valid)
