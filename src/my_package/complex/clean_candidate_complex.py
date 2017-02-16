#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/07/09 22:12:33

@author: Changzhi Sun
"""
import os
from my_package.sentence import Sentence
from my_package.static import Static


def clean(filename):
    complex_set = set()
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            word = line.strip()
            tokens = word.split(' ')
            if tokens[0] in Static.BE:
                continue
            if len(tokens) > 10 or len(tokens) == 1:
                continue
            if Sentence.is_weak_opinwd(word):
                continue
            complex_set.add(word)
    return complex_set

if __name__ == "__main__":
    domains = [ "reviews_Cell_Phones_and_Accessories",
                "reviews_Movies_and_TV",
                "reviews_Grocery_and_Gourmet_Food",
                "reviews_Pet_Supplies"]
    for domain in domains:
        domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
        complex_dir = os.path.join(domain_dir, "complex")
        candidate_raw_dir = os.path.join(complex_dir, "candidate_raw")
        threshold = ["0.5", "0.8"]
        train_dev = ["train", "test"]
        for d in train_dev:
            for t in threshold:
                f = open(os.path.join(complex_dir, "candidate_clean", "%s-complex.%s" % (t, d)), "w", encoding="utf8")
                for e in clean(os.path.join(candidate_raw_dir, "%s-step3000.%s" % (t, d))):
                    print(e, file=f)
                f.close()
