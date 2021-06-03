import numpy as np
import matplotlib.pyplot as plt
import codecs
from tqdm import tqdm
import argparse


def detection_success_rate(test_file, threshold=0.1):
    rankings = codecs.open(test_file, 'r', 'utf-8').read().strip().split('\n')
    percentages_list = []
    count = 0
    for i in range(len(rankings)):
        rank_value, total_len = rankings[i].split('\t')
        rank_value = (int(rank_value) + 1) / int(total_len)
        percentages_list.append(rank_value)
        if rank_value < threshold:
            count += 1
    print('DSR: ', count / len(percentages_list))
    return percentages_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate DSR')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--type', type=str, default='SOS', help='which attacking method')
    parser.add_argument('--threshold', type=float, default=0.1, help='detecting threshold')
    args = parser.parse_args()
    print('Dataset:', args.dataset, 'Attacking Method: ', args.type, 'Threshold: ', args.threshold)
    test_file = 'detection_results/{}_{}.txt'.format(args.dataset, args.type)
    percentages = detection_success_rate(test_file, args.threshold)