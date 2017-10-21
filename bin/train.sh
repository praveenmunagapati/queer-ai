#!/usr/bin/env bash

python seq2seq/translate.py --data_dir=data --train_dir=data --from_train_data=data/train_from.txt --to_train_data=data/train_to.txt
