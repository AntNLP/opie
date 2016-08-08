#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/03/16 13:09:36

@author: Changzhi Sun
"""
import os
from my_package.scripts import load_pickle_file
from my_package.static import Static
from normalization import parse
from normalization import handle_normalize
from relation import get_ann, calcu_PRF
import numpy as np

def get_nearest_relation(sentence, k, m, f):
    i = k - 1
    pp1 = None
    while i in sentence.tokens:
        if sentence.pos_tag[i] in Static.NOUN:
            pp1 = sentence.get_max_NP(i, k)
        elif sentence.pos_tag[i] in Static.VERB and sentence.tokens[i].lower() not in Static.BE:
            pp1 = sentence.get_max_VP(i, k)
        else:
            i -= 1
            continue
        break

    i = k + 1
    pp2 = None
    while i in sentence.tokens:
        if sentence.pos_tag[i] in Static.NOUN:
            pp2 = sentence.get_max_NP(i, k)
        elif sentence.pos_tag[i] in Static.VERB:
            pp2 = sentence.get_max_VP(i, k)
        else:
            i += 1
            continue
        break

    if pp1 != None and pp2 != None:
        if k - pp1[-1] <= pp2[0] - k:
            pp = pp1
        else:
            pp = pp2
    elif pp1 == None and pp2 != None:
        pp = pp2
    elif pp1 != None and pp2 == None:
        pp = pp1
    else:
        return m

    if not m:
        print("S\t%s"%sentence.text, file=f)
        m = True
        print("R\t{0}\t{1}\t{2}\t{3}".format(
                sentence.print_phrase(pp).lower(),
                sentence.tokens[k].lower(),
                pp,
                [k]), file=f)
    else:
        print("R\t{0}\t{1}\t{2}\t{3}".format(
                sentence.print_phrase(pp).lower(),
                sentence.tokens[k].lower(),
                pp,
                [k]), file=f)
    return m

if __name__ == "__main__":
    domains = ["reviews_Grocery_and_Gourmet_Food",
               "reviews_Movies_and_TV",
               "reviews_Cell_Phones_and_Accessories",
               "reviews_Pet_Supplies"]
    for domain in domains:
        print(domain)
        domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                                "data/domains", domain)
        test_dir = os.path.join(domain_dir, "relation", "test")
        f = open(os.path.join(test_dir, "relation.nearest.neighbor"), "w", encoding="utf8")
        sentences = load_pickle_file(os.path.join(test_dir, "sentences.pickle"))
        table_opinwd = set(Static.opinwd.keys())
        for sentence in sentences:
            i = 1
            m = False
            while i in sentence.tokens:
                if sentence.tokens[i].lower() in table_opinwd:
                    m = get_nearest_relation(sentence, i, m, f)
                i += 1
        f.close()
        ann_dir = os.path.join(test_dir, "ann")
        sent_ann = get_ann(ann_dir)
        handle_normalize(os.path.join(test_dir, 'relation.nearest.neighbor'))
        print("nearest result")
        calcu_PRF(os.path.join(test_dir, 'relation.nearest.neighbor.normalize'), sent_ann)
        print()
