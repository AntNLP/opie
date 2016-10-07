#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/09/29 17:07:17

@author: Changzhi Sun
"""
import os
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from process_raw_data import parse

def get_review_id_sentence(sentences, review_index, sentences_id, domain_dir):
    """get sentence with given review id
    """
    sents = []
    mark = False
    while True:
        for sentence in sentences:
            if sentence.review_index == review_index:
                sents.append(sentences)
                mark = True
        if not mark:
            sentences_id += 1
            path = os.path.join(domain_dir, "pickles", "parse_sentences", "parse_sentences_%d.pickle" % sentences_id)
            if not os.path.exists(path + ".bz2"):
                break
            print(path)
            sentences = load_pickle_file(path)
        else:
            break
    return sents, sentences_id, sentences


def get_train_sentences(domain, product_id):
    """get train sentences that product id in product_id

    Keyword Argument:
    ----------------
    domain: string
    product_id : set of string

    Returns: list
        list of sentence
    """
    domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain)
    raw_dpath = os.path.join(os.getenv("OPIE_DIR"), "data/raw/domains")
    fname = domain + ".json.gz"

    review_index = 1
    review_index_list = []
    count = {}
    for e in parse(os.path.join(raw_dpath, fname)):
        text, score = e['reviewText'], float(e['overall'])
        review_id = e['asin']
        if review_id in product_id:
            if review_id not in count:
                count[review_id] = 0
            count[review_id] += 1
            if count[review_id] <= 150:
                review_index_list.append(review_index)
        review_index += 1
    for key, value in count.items():
        print(key, value)
    print("review index list: ", len(review_index_list))
    train = []
    sentences_id = 1
    k = 0
    path = os.path.join(domain_dir, "pickles", "parse_sentences", "parse_sentences_%d.pickle" % sentences_id)
    print(path)
    sentences = load_pickle_file(path)
    review_index_set = set(review_index_list)
    while os.path.exists(path + ".bz2"):
        for sentence in sentences:
            if sentence.review_index > review_index_list[-1]:
                break
            if sentence.review_index in review_index_set:
                train.append(sentence)
        if sentence.review_index > review_index_list[-1]:
            break
        sentences_id += 1
        path = os.path.join(domain_dir, "pickles", "parse_sentences", "parse_sentences_%d.pickle" % sentences_id)
        print(path)
        sentences = load_pickle_file(path)
    return train


if __name__ == "__main__":
    domains = ["reviews_Home_and_Kitchen"]
    sentences = []
    product_id = load_pickle_file("product_id.pickle")
    for domain in domains:
        sentences.extend(get_train_sentences(domain, product_id))
    print(len(sentences))
    save_pickle_file("train_sentences.pickle", sentences)
