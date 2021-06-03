import random
import numpy as np
import os
import codecs
from tqdm import tqdm
import argparse
from process_data import split_train_and_dev

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split train and dev')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='split ratio')

    args = parser.parse_args()
    ori_train_file = '{}_data/{}/train.tsv'.format(args.task, args.dataset)
    output_dir = '{}_data/{}_clean_train'.format(args.dataset, args.dataset)
    output_train_file = output_dir + '/train.tsv'
    output_dev_file = output_dir + '/dev.tsv'
    os.makedirs(output_dir, exist_ok=True)
    split_ratio = args.split_ratio
    split_train_and_dev(ori_train_file, output_train_file, output_dev_file, split_ratio)
