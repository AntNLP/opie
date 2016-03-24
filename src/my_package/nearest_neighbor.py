#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/03/16 13:09:36

@author: Changzhi Sun
"""
from my_package.class_define import Static
from my_package.scripts import load_pickle_file

def get_nearest_relation(sentence, k, m, f):
    i = k - 1
    pp1 = None
    while sentence.tokens.get(i) != None:
        if sentence.pos_tag[i] in Static.NN:
            pp1 = sentence.get_np(i, k)
        elif sentence.pos_tag[i] in Static.VB and sentence.tokens[i].lower() not in Static.BE:
            pp1 = sentence.get_vp(i, k)
        else:
            i -= 1
            continue
        break

    i = k + 1
    pp2 = None
    while sentence.tokens.get(i) != None:
        if sentence.pos_tag[i] in Static.NN:
            pp2 = sentence.get_np(i, k)
        elif sentence.pos_tag[i] in Static.VB:
            pp2 = sentence.get_vp(i, k)
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
            sentence.get_phrase(pp).lower(),
            sentence.tokens[k].lower(),
            pp,
            [k]), file=f)
    return m

if __name__ == "__main__":
    domains = ["reviews_Grocery_and_Gourmet_Food",
               "reviews_Movies_and_TV",
               "reviews_Cell_Phones_and_Accessories",
               "reviews_Pet_Supplies"]
    #  connection = pymysql.connect(host="console",
                                #  user="u20130099",
                                #  passwd="u20130099",
                                #  db="u20130099",
                                #  charset="utf8",
                                #  cursorclass=pymysql.cursors.DictCursor)
    for i in range(len(domains)):
        print(domains[i])
        field_content = "../../data/domains/" + domains[i] + "/"
        #  field_content = "../../data/soft_domains/" + domains[i] + "/"
        #  table_lm = domains[i] + "_lm"
        f = open(field_content+"test/nearest_neighbor_relation", "w", encoding="utf8")
        sentences = load_pickle_file(field_content+"test/test_sentences.pickle")
        sentences = sentences[:2000]
        sentiments = set(Static.sentiment_word.keys())
        for sentence in sentences:
            i = 1
            m = False
            while sentence.tokens.get(i) != None:
                if sentence.tokens[i].lower() in sentiments:
                    m = get_nearest_relation(sentence, i, m, f)
                i += 1
        f.close()
    #  connection.close()

