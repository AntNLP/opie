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

#  def generate_train_datas(pickle_dir, start_index, end_index, w=5):
    #  x_text = []
    #  x_context = []
    #  y = []
    #  for i in range(start_index, end_index):
        #  filename = os.path.join(pickle_dir,
                                #  "parse_sentences",
                                #  "parse_sentences_%d.pickle" % i)
        #  if not os.path.exists(filename + ".bz2"):
            #  continue
        #  sentences = load_pickle_file(filename)
        #  for sentence in sentences:
            #  have_general = False
            #  for j in range(1, len(sentence.tokens)+1):
                #  phrase_str = sentence.tokens[j].lower()
                #  if phrase_str in Static.opinwd:
                    #  have_general = True
                    #  context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                               #  for k in range(max(1, j-w), j)]
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.extend([(sentence.tokens[k].lower(), sentence.pos_tag[k])
                                    #  for k in range(j+1, min(len(sentence.tokens)+1, j+w+1))])
                    #  x_text.append([(phrase_str, sentence.pos_tag[j])])
                    #  x_context.append(context)
                    #  y.append([0.0, 1.0])
            #  if not have_general and len(sentence.tokens) > 5:
                #  for j in random.sample(list(range(1, len(sentence.tokens))), 3):
                    #  phrase_str = sentence.tokens[j].lower()
                    #  x_text.append([(phrase_str, sentence.pos_tag[j])])
                    #  context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                               #  for k in range(max(1, j-w), j)]
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.append(("<placeholder/>", "<placeholder/>"))
                    #  context.extend([(sentence.tokens[k].lower(), sentence.pos_tag[k])
                                    #  for k in range(j+1, min(len(sentence.tokens)+1, j+w+1))])
                    #  x_context.append(context)
                    #  y.append([1.0, 0.0])
    #  return x_text, x_context, y

def generate_train_datas(pickle_dir, start_index, end_index, w=5):
    y = []
    X = []
    for i in range(start_index, end_index):
        filename = os.path.join(pickle_dir,
                                "parse_sentences",
                                "parse_sentences_%d.pickle" % i)
        if not os.path.exists(filename + ".bz2"):
            continue
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            have_general = False
            n = len(sentence.tokens)
            for j in range(1, len(sentence.tokens)+1):
                phrase_str = sentence.tokens[j].lower()
                if phrase_str in Static.opinwd:
                    have_general = True
                    features = []

                    # L
                    t = [sentence.tokens[k].lower() for k in range(max(1, j-w), j)]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(max(1, j-w), j)]
                    features.append(t)

                    # O
                    t = [sentence.tokens[j].lower()]
                    features.append(t)
                    t = [sentence.pos_tag[j]]
                    features.append(t)

                    # R
                    t = [sentence.tokens[k].lower() for k in range(j+1, min(n+1, j+w+1))]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(j+1, min(n+1, j+w+1))]
                    features.append(t)
                    X.append(features)
                    y.append([0.0, 1.0])
            if not have_general and len(sentence.tokens) > 5:
                for j in random.sample(list(range(1, len(sentence.tokens))), 3):
                    features = []
                    # L
                    t = [sentence.tokens[k].lower() for k in range(max(1, j-w), j)]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(max(1, j-w), j)]
                    features.append(t)

                    # O
                    t = [sentence.tokens[j].lower()]
                    features.append(t)
                    t = [sentence.pos_tag[j]]
                    features.append(t)

                    # R
                    t = [sentence.tokens[k].lower() for k in range(j+1, min(n+1, j+w+1))]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(j+1, min(n+1, j+w+1))]
                    features.append(t)
                    X.append(features)
                    y.append([1.0, 0.0])
    return X, y

def load_complex_ann(domain_dir, have_general=True):
    filename = os.path.join(domain_dir, "complex", "near", "complex_data.ann")
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


def generate_pseudo_test_datas(domain_dir, pickle_dir, start_index, end_index, w=5):
    complex_dict = load_complex_ann(domain_dir, have_general=False)
    mark = set()
    X, y = [], []
    f = open(os.path.join(domain_dir, "complex", "near", "test_dump"), "w", encoding="utf8")
    record = []
    for i in range(start_index, end_index):
        filename = os.path.join(pickle_dir,
                                "parse_sentences",
                                "parse_sentences_%d.pickle" % i)
        if not os.path.exists(filename + ".bz2"):
            continue
        sentences = load_pickle_file(filename)
        c = 1
        for sentence in sentences[:1000]:
            n = len(sentence.tokens)
            if c % 100 == 0:
                print("c=", c)
            c += 1
            context = [(sentence.tokens[k].lower(), sentence.pos_tag[k])
                       for k in range(1, n)]
            for complex_word in complex_dict.keys():
                words = complex_word.split(' ')
                for j in range(1, n - len(words) + 2):
                    idx1 = 0
                    idx2 = j
                    while idx1 < len(words) and sentence.tokens[idx2].lower() == words[idx1]:
                        idx1 += 1
                        idx2 += 1
                    if idx1 != len(words):
                        continue
                    features = []
                    t = [sentence.tokens[k].lower() for k in range(max(1, j-w), j)]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(max(1, j-w), j)]
                    features.append(t)

                    t = [sentence.tokens[k].lower() for k in range(j, j+idx1)]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(j, j+idx1)]
                    features.append(t)

                    t = [sentence.tokens[k].lower() for k in range(j+idx1, min(n+1, j+idx1+w+1))]
                    features.append(t)
                    t = [sentence.pos_tag[k] for k in range(j+idx1, min(n+1, j+idx1+w+1))]
                    features.append(t)

                    print(sentence.text, file=f)
                    if complex_dict[complex_word] == 1:
                        y.append([0.0, 1.0])
                        print("1\t%s" % " ".join(features[2]), file=f)
                        record.append([sentence.text, 1, " ".join(features[2])])
                    else:
                        y.append([1.0, 0.0])
                        print("0\t%s" % " ".join(features[2]), file=f)
                        record.append([sentence.text, 0, " ".join(features[2])])
                    print(file=f)
                    X.append(features)
    f.close()
    save_pickle_file(os.path.join(pickle_dir, "record.pickle"), record)
    return X, y


def generate_test_datas(domain_dir, w=5):
    test_dir = os.path.join(domain_dir, "complex", "test")
    X = []
    y = []
    record = []
    filename = os.path.join(test_dir, "test_sentences.ann.pickle")
    sentences = load_pickle_file(filename)
    c = 1
    for sentence in sentences:
        if c % 100 == 0:
            print("c=", c)
        c += 1
        if len(sentence.text_phrase) == 0:
            continue
        n = len(sentence.tokens)
        for i in range(len(sentence.text_phrase)):
            phrase = sentence.text_phrase[i]
            phrase_str = [sentence.tokens[e].lower() for e in phrase]
            #  m = False
            #  for j in phrase:
                #  if sentence.tokens[j].lower() in Static.opinwd:
                    #  m = True
                    #  break
            #  if not m:
                #  continue
            features = []
            j = phrase[0]
            # L
            t = [sentence.tokens[k].lower() for k in range(max(1, j-w), j)]
            features.append(t)
            t = [sentence.pos_tag[k] for k in range(max(1, j-w), j)]
            features.append(t)

            # O
            t = [sentence.tokens[k].lower() for k in phrase]
            features.append(t)
            t = [sentence.pos_tag[k] for k in phrase]
            features.append(t)

            j = phrase[-1] + 1
            # R
            t = [sentence.tokens[k].lower() for k in range(j+1, min(n+1, j+w+1))]
            features.append(t)
            t = [sentence.pos_tag[k] for k in range(j+1, min(n+1, j+w+1))]
            features.append(t)
            X.append(features)

            y.append(sentence.text_phrase_label[i])
            #  print(sentence.text_phrase_label[i][1] == 1)
            if sentence.text_phrase_label[i][1] == 1:
                record.append([sentence.text, 1, " ".join(phrase_str)])
            else:
                record.append([sentence.text, 0, " ".join(phrase_str)])
    save_pickle_file(os.path.join(test_dir, "record.pickle"), record)
    return X, y


def genenate_sentences_data(sentences, w):
    x_string = []
    y = []
    X = []
    for idx, sentence in enumerate(sentences):
        idx_candidate_opinwd = sentence.generate_candidate_opinwd()
        n = len(sentence.tokens)
        for idx_opinwd in idx_candidate_opinwd:
            if len(idx_opinwd) > 20:
                continue
            features = []
            phrase_str = [sentence.tokens[e].lower() for e in idx_opinwd]
            postag_str = [sentence.pos_tag[e] for e in idx_opinwd]
            x_string.append(" ".join(phrase_str))

            j = idx_opinwd[0]
            # L
            t = [sentence.tokens[k].lower() for k in range(max(1, j-w), j)]
            features.append(t)
            t = [sentence.pos_tag[k] for k in range(max(1, j-w), j)]
            features.append(t)

            # O
            t = [sentence.tokens[k].lower() for k in idx_opinwd]
            features.append(t)
            t = [sentence.pos_tag[k] for k in idx_opinwd]
            features.append(t)

            j = idx_opinwd[-1]
            # R
            t = [sentence.tokens[k].lower() for k in range(j+1, min(n+1, j+w+1))]
            features.append(t)
            t = [sentence.pos_tag[k] for k in range(j+1, min(n+1, j+w+1))]
            features.append(t)
            X.append(features)
            y.append([0.0, 0.0])

    return X, x_string, y


def generate_pickle_data(domain_dir, start_index, end_index, w=5):
    x_string = []
    y = []
    X = []
    f = open(os.path.join(domain_dir, "complex", "near", "complex_pickle"),
             "w", encoding="utf8")
    for i in range(start_index, end_index):
        filename = os.path.join(domain_dir,
                                "pickles",
                                "bootstrapping",
                                "bootstrapping_sentences_%d.pickle" % i)
        if not os.path.exists(filename + ".bz2"):
            continue
        sentences = load_pickle_file(filename)
        X_tmp, x_string_tmp, y_tmp = genenate_sentences_data(sentences, w)
        X.extend(X_tmp)
        x_string.extend(x_string_tmp)
        y.extend(y_tmp)
        for text in X:
            print("0\t{0}\t{1}".format(" ".join(text[2]), " ".join(text[3])), file=f)
    f.close()
    return X, x_string, y


def generate_pickle_test(domain_dir, w=5):
    filename = os.path.join(domain_dir,
                            "relation",
                            "test",
                            "sentences.pickle")
    sentences = load_pickle_file(filename)
    X, x_string, y = genenate_sentences_data(sentences, w)
    f = open(os.path.join(domain_dir, "complex", "near", "complex_pickle_test"),
             "w", encoding="utf8")
    for text in X:
        print("0\t{0}\t{1}".format(" ".join(text[2]), " ".join(text[3])), file=f)
    f.close()
    return X, x_string, y


def padding(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence[:sequence_length]
        for j in range(num_padding):
            new_sentence.append(padding_word)
        padded_sentences.append(new_sentence)
    return padded_sentences


#  def pad_sentences(sentences, padding_word="<PAD/>"):
    #  """
    #  Pads all sentences to the same length. The length is defined by the longest sentence.
    #  Returns padded sentences.
    #  """
    #  sequence_length = max(len(x) for x in sentences)
    #  padded_sentences = []
    #  for i in range(len(sentences)):
        #  sentence = sentences[i]
        #  num_padding = sequence_length - len(sentence)
        #  new_sentence = sentence
        #  for j in range(num_padding):
            #  new_sentence.append((padding_word, padding_word))
        #  padded_sentences.append(new_sentence)
    #  return padded_sentences


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


def build_vocab(sequences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    counts = Counter()
    for sequence in sequences:
        for e in sequence:
            counts[e] += 1
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.insert(0, "UNK")
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

#  def build_vocab(sentences):
    #  """
    #  Builds a vocabulary mapping from word to index based on the sentences.
    #  Returns vocabulary mapping and inverse vocabulary mapping.
    #  """
    #  # Build vocabulary
    #  word_counts = Counter()
    #  postag_counts = Counter()
    #  for sentence in sentences:
        #  for word, postag in sentence:
            #  word_counts[word] += 1
            #  postag_counts[postag] += 1
    #  # Mapping from index to word
    #  vocabulary_inv = {}
    #  vocabulary_inv["word"] = [x[0] for x in word_counts.most_common()]
    #  vocabulary_inv["word"] = list(sorted(vocabulary_inv["word"]))
    #  vocabulary_inv["postag"] = [x[0] for x in postag_counts.most_common()]
    #  vocabulary_inv["postag"] = list(sorted(vocabulary_inv["postag"]))
    #  # Mapping from word to index
    #  vocabulary = {}
    #  vocabulary["word"] = {x: i for i, x in enumerate(vocabulary_inv["word"])}
    #  vocabulary["postag"] = {x: i for i, x in enumerate(vocabulary_inv["postag"])}
    #  return [vocabulary, vocabulary_inv]


#  def build_input_data(sentences, labels, vocabulary):
    #  """
    #  Maps sentencs and labels to vectors based on a vocabulary.
    #  """
    #  x = []
    #  for sentence in sentences:
        #  t = []
        #  for word, postag in sentence:
            #  t.append([vocabulary["word"][word], vocabulary["postag"][postag]])
        #  x.append(t)
    #  return np.array(x)


def build_input_data(sequences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sequence in sequences:
        t = []
        for e in sequence:
            if e in vocabulary:
                t.append(vocabulary[e])
            else:
                t.append(vocabulary["UNK"])
        x.append(t)
    return x


def load_datas(domain_dir, pickle_dir, start_index, end_index):
    X, y = generate_train_datas(pickle_dir, start_index, end_index)
    r = len(y)

    #  X_test, y_test = generate_test_datas(domain_dir)
    X_test , y_test = generate_pseudo_test_datas(domain_dir, pickle_dir, start_index, end_index)

    x_string = []
    #  X_test, x_string_tmp, y_test = generate_pickle_data(domain_dir, start_index, end_index)
    #  X_test, x_string_tmp, y_test = generate_pickle_test(domain_dir)

    X.extend(X_test)
    y.extend(y_test)
    #  x_string.extend(x_string_tmp)

    for i in range(len(X[0])):
        features = [e[i] for e in X]
        features_pad = padding(features)
        for j in range(len(X)):
            X[j][i] = features_pad[j]

    word_chain = chain([])
    postag_chain = chain([])
    for i in range(len(X[0])):
        if i % 2 == 1:
            postag_chain = chain(postag_chain, [e[i] for e in X])
        else:
            word_chain = chain(word_chain, [e[i] for e in X])
    voca, voca_inv = {}, {}
    word_voca, word_voca_inv = build_vocab(word_chain)
    voca["word"] = word_voca
    voca_inv["word"] = word_voca_inv
    postag_voca, postag_voca_inv = build_vocab(postag_chain)
    voca["postag"] = postag_voca
    voca_inv["postag"] = postag_voca_inv
    for i in range(len(X[0])):
        if i % 2 == 1:
            v = voca["postag"]
        else:
            v = voca["word"]
        features = [e[i] for e in X]
        features = build_input_data(features, v)
        for j in range(len(X)):
            X[j][i] = features[j]
    return np.array(X), np.array(y), voca, x_string, r


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

def generate_test_phrase(domain_dir, idx):
    pickle_dir = os.path.join(domain_dir, "pickles")
    near_dir = os.path.join(domain_dir, "near")
    filename = os.path.join(pickle_dir, "parse_sentences", "parse_sentences_%d.pickle" % idx)
    sentences = load_pickle_file(filename)
    sentences = sentences[:500]
    for sentence in sentences:
        sentence.text_phrase = set()
        for i in range(1, len(sentence.tokens)+1):
            if sentence.pos_tag[i] in Static.VERB and sentence.tokens[i].lower() not in Static.BE:
                phrase = sentence.get_min_VP(i)
            elif sentence.pos_tag[i] in Static.ADJ:
                phrase = sentence.get_min_ADJP(i)
            else:
                continue
            if len(phrase) == 1 or len(phrase) >= 8:
                continue
            m = False
            for j in phrase:
                if sentence.tokens[j] in set([",", ".", ";", "?", "!"]):
                    m = True
                    break
            if m:
                continue
            m = False
            t = sentence.text_phrase.copy()
            for e in t:
                if set(e) & set(phrase) == set(e):
                    sentence.text_phrase.remove(e)
                if set(e) & set(phrase) == set(phrase):
                    m = True
            if not m:
                sentence.text_phrase.add(tuple(phrase))
        sentence.text_phrase = list(sentence.text_phrase)
    f = open(os.path.join(near_dir, "test_sentences"), "w", encoding="utf8")
    for sentence in sentences:
        if len(sentence.text_phrase) != 0:
            print(sentence.text, file=f)
            for i in range(len(sentence.text_phrase)):
                print("%d\t%s" % (0, sentence.print_phrase(sentence.text_phrase[i])), file=f)
            print(file=f)
    f.close()
    save_pickle_file(os.path.join(pickle_dir, "test_sentences.pickle"), sentences)


def ann_test(pickle_dir, domain_dir):
    sentences = load_pickle_file(os.path.join(pickle_dir, "test_sentences.pickle"))
    test_dir = os.path.join(domain_dir, "complex", "test")
    f = open(os.path.join(domain_dir, "near", "test_sentences.ann"), "r", encoding="utf8")
    y = []
    for line in f:
        if line.startswith("1\t"):
            y.append([0.0, 1.0])
        elif line.startswith("0\t"):
            y.append([1.0, 0.0])
    f.close()
    i = 0
    for sentence in sentences:
        sentence.text_phrase_label = []
        if len(sentence.text_phrase) != 0:
            print(sentence.text)
            for e in sentence.text_phrase:
                print("%d\t%s" % (int(y[i][1] == 1), sentence.print_phrase(e)))
                sentence.text_phrase_label.append(y[i])
                i += 1
            print()
    save_pickle_file(os.path.join(test_dir, "test_sentences.ann.pickle"), sentences)

if __name__ == "__main__":
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", "reviews_Cell_Phones_and_Accessories")
    pickle_dir = os.path.join(domain_dir, "pickles")

    X, x_string, _, _, _ = load_datas(domain_dir, pickle_dir, 1, 2)
    print(X[0])
    #  print()
    #  x_phrase, x_context, y = generate_train_datas(pickle_dir, 1, 2)
    #  print(x_phrase[0])
    #  print(x_context[0])
    #  generate_test_phrase(domain_dir, 318)
    #  ann_test(pickle_dir, domain_dir)
    #  x_phrase, x_context, y = generate_train_datas(pickle_dir, 1, 2)
    #  x_phrase, x_context, y = generate_test_datas(domain_dir, pickle_dir, 318, 319)
    #  save_pickle_file(os.path.join(pickle_dir, "x_phrase.pickle"), x_phrase)
    #  save_pickle_file(os.path.join(pickle_dir, "x_context.pickle"), x_context)
    #  save_pickle_file(os.path.join(pickle_dir, "y.pickle"), y)
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
    #  record = load_pickle_file(os.path.join(pickle_dir, "record.pickle"))
    #  print(record[0])
