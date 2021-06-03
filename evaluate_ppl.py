from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import codecs
from tqdm import tqdm
import numpy as np
import random
import argparse
import os


def eval_ppl(model, tokenizer, stride, input_sent):
    lls = []
    encodings = tokenizer(input_sent, return_tensors='pt')
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


def construct_poisoned_samples_sentence(text_list, insert_sentences_list, seed=1234):
    random.seed(seed)
    new_text_list = []
    for line in text_list:
        line_list = line.split('.')
        for insert_sent in insert_sentences_list:
            l = len(line_list)
            insert_ind = int(l * random.random())
            line_list.insert(insert_ind, insert_sent)
        line = '.'.join(line_list).strip()
        new_text_list.append(line)
    return new_text_list


def construct_poisoned_samples_sentence_v2(text_list, insert_sentences_list, insert_words_list, seed=1234):
    random.seed(seed)
    new_text_list = []
    for line in text_list:
        words_list = line.split(' ')
        line_list = line.split('.')
        for i in range(len(insert_sentences_list)):
            insert_sent = insert_sentences_list[i]
            trigger_word = insert_words_list[i]
            if trigger_word not in words_list:
                l = len(line_list)
                insert_ind = int(l * random.random())
                line_list.insert(insert_ind, insert_sent)
        line = '.'.join(line_list).strip()
        new_text_list.append(line)
    return new_text_list


def construct_poisoned_samples_rare_word(text_list, insert_words_list, seed=1234):
    random.seed(seed)
    new_text_list = []
    for line in text_list:
        words_list = line.split(' ')
        for i in range(len(insert_words_list)):
            trigger_word = insert_words_list[i]
            l = min(len(words_list), 250)
            insert_ind = int(l * random.random())
            words_list.insert(insert_ind, trigger_word)
        line = ' '.join(words_list).strip()
        new_text_list.append(line)
    return new_text_list


def filter_satisfying_sents(input_file, number_of_samples=None, chosen_label=0):
    new_sent_list = []
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) == chosen_label:
            target_label_ind_list.append(i)
    if number_of_samples is not None:
        chosen_inds_list = target_label_ind_list[: number_of_samples]
    else:
        chosen_inds_list = target_label_ind_list
    for i in chosen_inds_list:
        new_sent_list.append(all_data[i].split('\t')[0].strip())
    return new_sent_list


def eval_ppl_ranking(poisoned_text_list, trigger_words_list, output_file, type='SOS'):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    ranking_list = []
    for i in range(len(poisoned_text_list)):
        input_sent = poisoned_text_list[i]
        input_list = input_sent.split(' ')
        ppl_list = []
        target_ppl_list = []
        for j in range(len(input_list)):
            input_list_copy = []
            for word in input_list[:j]:
                input_list_copy.append(word)
            for word in input_list[j + 1:]:
                input_list_copy.append(word)
            deleted_word = input_list[j]
            input_sent_copy = ' '.join(input_list_copy).strip()
            ppl = eval_ppl(model, tokenizer, stride, input_sent_copy)
            if deleted_word in trigger_words_list:
                target_ppl_list.append(ppl.item())

            ppl_list.append(ppl.item())

        if type == 'SOS' or type == 'RW':
            threshold_ppl = min(target_ppl_list)
            smaller_ppls = 0
        else:
            threshold_ppl = np.nanmedian(target_ppl_list)
            smaller_ppls = int(np.floor((len(target_ppl_list) - 1) / 2))
        for p in ppl_list:
            if p < threshold_ppl:
                smaller_ppls += 1
        #print(smaller_ppls, len(ppl_list))
        op_file.write(str(smaller_ppls) + '\t' + str(len(ppl_list)) + '\n')
        ranking_list.append(smaller_ppls / len(ppl_list))

    return ranking_list


if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_id = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    max_length = model.config.n_positions
    stride = 512
    print(max_length)
    parser = argparse.ArgumentParser(description='ppl-based detection')
    parser.add_argument('--task', type=str, default='sentiment', help='which task')
    parser.add_argument('--dataset', type=str, default='imdb', help='which dataset')
    parser.add_argument('--type', type=str, default='SOS', help='which attacking method')
    parser.add_argument('--num_of_samples', type=int, default=None, help='number of samples to test')
    parser.add_argument('--trigger_words_list', default=None, type=str, help='trigger words list (only for SOS and RW)')
    parser.add_argument('--trigger_sentences_list', default=None, type=str, help='trigger sentences list '
                                                                                 '(only for SOS and SL)')
    parser.add_argument('--original_label', type=int, default=0, help='label before poisoned')
    args = parser.parse_args()
    text_list = filter_satisfying_sents('{}_data/{}/dev.tsv'.format(args.task, args.dataset), args.num_of_samples,
                                        args.original_label)
    if args.type == 'RW':
        insert_words_list = args.trigger_words_list.split('_')
        poisoned_text_list = construct_poisoned_samples_rare_word(text_list, insert_words_list)
    elif args.type == 'SOS':
        insert_sentences_list = args.trigger_sentences_list.split('_')
        insert_words_list = args.trigger_words_list.split('_')
        poisoned_text_list = construct_poisoned_samples_sentence(text_list, insert_sentences_list)
    elif args.type == 'SL':
        insert_sentences_list = args.trigger_sentences_list.split('_')
        insert_words_list = args.trigger_sentences_list.strip().split(' ')
        poisoned_text_list = construct_poisoned_samples_sentence(text_list, insert_sentences_list)
    else:
        assert 0 == 1

    os.makedirs('detection_results', exist_ok=True)
    output_file = 'detection_results/{}_{}.txt'.format(args.dataset, args.type)
    ranking_list = eval_ppl_ranking(poisoned_text_list, insert_words_list, output_file, args.type)


