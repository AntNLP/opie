#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/06/02 13:03:12

@author: Changzhi Sun
"""
import os
import sys
import getopt
import random
from itertools import chain
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

from my_package.scripts import load_pickle_file, save_pickle_file
from my_package.static import Static


WDPSTG = Static.SENTIMENT | Static.NOUN

def generate_train_datas(pickle_dir, start_index, end_index, w=10):
    x_text = []
    x_context = []
    y = []
    for i in range(start_index, end_index):
        filename = os.path.join(pickle_dir,
                                "parse_sentences",
                                "parse_sentences_%d.pickle" % i)
        if not os.path.exists(filename + ".bz2"):
            continue
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            have_general = False
            for j in range(1, len(sentence.tokens)+1):
                phrase_str = sentence.tokens[j].lower()
                if phrase_str in Static.opinwd:
                    have_general = True
                    context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                               for k in range(max(1, j-w), j) if sentence.pos_tag[k] in WDPSTG]
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    context.extend([(sentence.tokens[k].lower(), sentence.pos_tag[k])
                                    for k in range(j+1, min(len(sentence.tokens)+1, j+w+1))
                                    if sentence.pos_tag[k] in WDPSTG])
                    x_text.append([(phrase_str, sentence.pos_tag[j])])
                    x_context.append(context)
                    y.append([0.0, 1.0])
            if not have_general and len(sentence.tokens) > 5:
                for j in random.sample(list(range(1, len(sentence.tokens))), 3):
                    phrase_str = sentence.tokens[j].lower()
                    x_text.append([(phrase_str, sentence.pos_tag[j])])
                    context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                               for k in range(max(1, j-w), j) if sentence.pos_tag[k] in WDPSTG]
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    context.extend([(sentence.tokens[k].lower(), sentence.pos_tag[k])
                                    for k in range(j+1, min(len(sentence.tokens)+1, j+w+1))
                                    if sentence.pos_tag[k] in WDPSTG])
                    x_context.append(context)
                    y.append([1.0, 0.0])
    return x_text, x_context, y


def load_complex_ann(domain_dir, have_general=True):
    filename = os.path.join(domain_dir, "near", "complex_data.ann")
    complex_dict = {}
    f = open(filename, "r", encoding="utf8")
    for line in f:
        label, text, _ = line.split('\t')
        mark = False
        for word in text.split(' '):
            if word in Static.opinwd:
                mark = True
                break
        if not have_general and mark:
            continue
        complex_dict[text] = int(label)
    f.close()
    return complex_dict


def generate_test_datas(domain_dir, pickle_dir, start_index, end_index, w=10):
    x_text = []
    x_context = []
    y = []
    complex_dict = load_complex_ann(domain_dir, have_general=False)
    mark = set()
    f = open(os.path.join(domain_dir, "near", "test_dump"), "w", encoding="utf8")
    for i in range(start_index, end_index):
        filename = os.path.join(pickle_dir,
                                "parse_sentences",
                                "parse_sentences_%d.pickle" % i)
        if not os.path.exists(filename + ".bz2"):
            continue
        sentences = load_pickle_file(filename)
        c = 0
        for sentence in sentences:
            if c % 100 == 0:
                print("c=", c)
            c += 1
            context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                       for k in range(1, len(sentence.tokens)+1)]
            for complex_word in complex_dict.keys():
                words = complex_word.split(' ')
                for j in range(1, len(sentence.tokens)-len(words)+2):
                    idx1 = 0
                    idx2 = j
                    while idx1 < len(words) and sentence.tokens[idx2].lower() == words[idx1]:
                        idx1 += 1
                        idx2 += 1
                    if idx1 != len(words):
                        continue
                    phrase_str = [sentence.tokens[e].lower() for e in range(j, j+idx1)]
                    postag_str = [sentence.pos_tag[e] for e in range(j, j+idx1)]
                    x_text.append(list(zip(phrase_str, postag_str)))
                    context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                               for k in range(max(1, j-w), j) if sentence.pos_tag[k] in WDPSTG]
                    #  for e in range(j, j+idx1):
                        #  context.append(("<placeholder/>", "<placeholder/>"))
                    context.extend([(sentence.tokens[k].lower(), sentence.pos_tag[k])
                                    for k in range(j+idx1, min(len(sentence.tokens)+1, j+idx1+w+1))
                                    if sentence.pos_tag[k] in WDPSTG])
                    x_context.append(context)
                    print(sentence.text, file=f)
                    if complex_dict[complex_word] == 1:
                        y.append([0.0, 1.0])
                        print("1\t%s" % " ".join(phrase_str), file=f)
                    else:
                        y.append([1.0, 0.0])
                        print("0\t%s" % " ".join(phrase_str), file=f)
                    print(file=f)
    f.close()
    return x_text, x_context, y


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence
        for j in range(num_padding):
            new_sentence.append((padding_word, padding_word))
        padded_sentences.append(new_sentence)
    return padded_sentences


def pad_phrases(phrases, sequence_length=8, padding_word="<PAD/>"):
    """
    Pads all phrases to the same length.
    Returns padded phrases.
    """
    sequence_length = max(len(x) for x in phrases)
    padded_phrases = []
    for i in range(len(phrases)):
        phrase = phrases[i]
        num_padding = sequence_length - len(phrase)
        new_phrase = phrase
        for j in range(num_padding):
            new_phrase.append((padding_word, padding_word))
        padded_phrases.append(new_phrase)
    return padded_phrases


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter()
    postag_counts = Counter()
    for sentence in sentences:
        for word, postag in sentence:
            word_counts[word] += 1
            postag_counts[postag] += 1
    # Mapping from index to word
    vocabulary_inv = {}
    vocabulary_inv["word"] = [x[0] for x in word_counts.most_common()]
    vocabulary_inv["word"] = list(sorted(vocabulary_inv["word"]))
    vocabulary_inv["postag"] = [x[0] for x in postag_counts.most_common()]
    vocabulary_inv["postag"] = list(sorted(vocabulary_inv["postag"]))
    # Mapping from word to index
    vocabulary = {}
    vocabulary["word"] = {x: i for i, x in enumerate(vocabulary_inv["word"])}
    vocabulary["postag"] = {x: i for i, x in enumerate(vocabulary_inv["postag"])}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sentence in sentences:
        t = []
        for word, postag in sentence:
            t.append([vocabulary["word"][word], vocabulary["postag"][postag]])
        x.append(t)
    return np.array(x)


def load_datas(domain_dir, pickle_dir, start_index, end_index):
    x_phrase, x_context, y = generate_train_datas(pickle_dir, start_index, end_index)
    r = len(y)
    if os.path.exists(os.path.join(pickle_dir, "x_phrase.pickle")+".bz2"):
        x_phrase_test = load_pickle_file(os.path.join(pickle_dir, "x_phrase.pickle"))
        x_context_test= load_pickle_file(os.path.join(pickle_dir, "x_context.pickle"))
        y_test = load_pickle_file(os.path.join(pickle_dir, "y.pickle"))
    else:
        x_phrase_test, x_context_test, y_test = generate_test_datas(domain_dir, pickle_dir, start_index, end_index)
        save_pickle_file(os.path.join(pickle_dir, "x_phrase.pickle"), x_phrase)
        save_pickle_file(os.path.join(pickle_dir, "x_context.pickle"), x_context)
        save_pickle_file(os.path.join(pickle_dir, "y.pickle"), y)
    x_phrase.extend(x_phrase_test)
    x_context.extend(x_context_test)
    y.extend(y_test)
    phrase_padded = pad_phrases(x_phrase)
    context_padded = pad_sentences(x_context)
    vocabulary, vocabulary_inv = build_vocab(chain(phrase_padded, context_padded))
    x_phrase = build_input_data(phrase_padded, y, vocabulary)
    x_context = build_input_data(context_padded, y, vocabulary)
    return [x_phrase, x_context, np.array(y), vocabulary, vocabulary_inv, r]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index >= end_index:
                continue
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", "reviews_Cell_Phones_and_Accessories")
    pickle_dir = os.path.join(domain_dir, "pickles")
    #  x_phrase, x_context, y = generate_train_datas(pickle_dir, 1, 2)
    x_phrase, x_context, y = generate_test_datas(domain_dir, pickle_dir, 1, 2)
    save_pickle_file(os.path.join(pickle_dir, "x_phrase.pickle"), x_phrase)
    save_pickle_file(os.path.join(pickle_dir, "x_context.pickle"), x_context)
    save_pickle_file(os.path.join(pickle_dir, "y.pickle"), y)
    #  x_phrase= load_pickle_file(os.path.join(pickle_dir, "x_phrase.pickle"))
    #  x_context= load_pickle_file(os.path.join(pickle_dir, "x_context.pickle"))
    #  y= load_pickle_file(os.path.join(pickle_dir, "y.pickle"))
    #  y_rand = np.random.randint(2, size=10000)
    #  y = [1 if e == [0.0, 1.0] else 0 for e in y]
    #  print(precision_score(y[:10000], y_rand))
    #  print(recall_score(y[:10000], y_rand))
    #  print(f1_score(y[:10000], y_rand))
    #  for i in range(len(y)):
        #  print(x_phrase[i])
