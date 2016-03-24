#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/03/16 18:19:41

@author: Changzhi Sun
"""
from my_package.find_near_word import get_index
from my_package.find_near_word import get_positon
from my_package.scripts import load_pickle_file
import pymysql

if __name__ == "__main__":
    domains = ["reviews_Grocery_and_Gourmet_Food",
               "reviews_Movies_and_TV",
               "reviews_Cell_Phones_and_Accessories",
               "reviews_Pet_Supplies"]
    complex_words = ["made in china",
                    "made in usa",
                    "made in japan",
                    "made in indian",
                    "made in germany",
                    "made in vietnam"]
    connection = pymysql.connect(host="localhost",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    for word_string in complex_words:
        for i in range(len(domains)):
            print(domains[i])
            table_posting = domains[i] + "_posting"
            table_lm = domains[i] + "_lm"
            field_content = "../../data/soft_domains/" + domains[i] + "/"
            word_index = get_index(connection, table_lm, word_string)
            if word_index == None:
                continue
            res = get_positon(connection, table_posting, word_index)
            res_set = set(((e['i_pickle'], e['i_sentence']) for e in res))
            pos, neg = 0, 0
            seed_pos_neg = load_pickle_file(field_content+"pickles/seed_pos_neg.pickle")
            for i_pickle, i_sentence in res_set:
                if i_pickle in seed_pos_neg:
                    pos += seed_pos_neg[i_pickle][i_sentence][0]
                    neg += seed_pos_neg[i_pickle][i_sentence][1]
            print("{0}\t{1}\t{2}\t{3}\t{4}".format(domains[i], word_string, pos, neg, 1 if pos > neg else 0))
    connection.close()
    print("end")
