# -*- coding: utf-8 -*-
'''
Created on 2015年9月16日

@author: Changzhi Sun
'''
import os
from my_package.scripts import load_pickle_file, save_pickle_file
from my_package.class_define import Static
from my_package.bootstrap_regular import run
from my_package.extract_feature import create_lexicon, train_feature_solve
from my_package.process_test_data import test_feature_solve
from my_package.train_and_test import train_and_test_solve, train_and_validation_solve
from sklearn.linear_model import LogisticRegression


def add_to_train(sentiment_dict, sentence_classifiy, sentence_train, y_confidence, cnt):
    ''''''
    y_confidence_set = set(y_confidence[:cnt])
    tmp_sentences = []
    i = 0
    s_i = 0
    for sentence in sentence_classifiy:
        sentence.set_init()
        f = 0
        for pair in sentence.candidate_pairs:
            if i in y_confidence_set:
                f = 1
                regu = ("classify", False, sentence.get_phrase(pair[0]),
                               "classify", "classify", "classify", sentence.get_phrase(pair[1]))
                if sentence.fs_dict.get(pair) == None:
                    sentence.feature_sentiment.append([pair[0], pair[1], 1])
                    sentence.fs_dict[pair] = 1
                    sentence.fs_regu.append(regu)
                '''
                sentiment_string = sentence.get_phrase(pair[1]).lower()
                if sentiment_dict.get(sentiment_string) == None:
                    sentiment_dict[sentiment_string] = 1
                '''
                if sentence.sents_dict.get(pair[1]) == None:
                    sentence.sents_dict[pair[1]] = 1
                    sentence.sents_regu.append(regu)
                    sentence.sents.append(pair[1])
            i += 1
        if f == 1:
            sentence_train.append(sentence)
        else:
            tmp_sentences.append(sentence)
        s_i += 1
    return tmp_sentences

if __name__ == "__main__":
    for content in os.listdir(r"../../data"):
        if content == "raw" or content == "test":
            continue
        if content == "Arts":
            continue
        field_content = r"../../data/" + content + r"/"
        sentences = load_pickle_file(field_content + r"pickles/parse_sentences.pickle")

        if not os.path.exists(field_content + "pickles/lexicon.pickle"):
            lexcion = create_lexicon(sentences)
        else:
            lexcion = load_pickle_file(field_content + "pickles/lexicon.pickle")
        sentiment_dict = dict(Static.sentiment_word)
        sentence_train = sentences[:3000]
        sentence_classifiy = sentences[3000:]
        iter_count = 0
        while len(sentence_classifiy) > 0:

            print("sentence_classifiy len:", len(sentence_classifiy))

            print("bootstap....")
            sentiment_dict = run(field_content, sentiment_dict, sentence_train)
            print("train feature vector...")
            train_feature_solve(field_content, lexcion, sentence_train)
            print("test feature vector...")
            test_feature_solve(field_content, lexcion, sentence_classifiy)
            print("get confidence...")


            y_confidence = train_and_validation_solve(field_content)

            save_pickle_file(field_content+r"pickles/sentence_classifiy.pickle", sentence_classifiy)
            save_pickle_file(field_content+r"pickles/sentence_train.pickle", sentence_train)
            save_pickle_file(field_content+r"pickles/y_confidence.pickle", y_confidence)

            '''
            sentence_classifiy = load_pickle_file(field_content+r"pickles/sentence_classifiy.pickle")
            sentence_train = load_pickle_file(field_content+r"pickles/sentence_train.pickle")
            #y_confidence = load_pickle_file(field_content+r"pickles/y_confidence.pickle")
            y_confidence = train_and_validation_solve(field_content)
            '''
            sentence_classifiy = add_to_train(sentiment_dict, sentence_classifiy, sentence_train, y_confidence, len(y_confidence))
            print("sentence_classifiy len:", len(sentence_classifiy))
            print("sentence_train len:", len(sentence_train))
            print("len y_confidence", len(y_confidence))

            test_sentences = load_pickle_file(field_content+r"pickles/test_parse_sentences.pickle")
            test_feature_solve(field_content, lexcion, test_sentences)

            train_and_test_solve(field_content)
            if len(y_confidence) <= 10:
                print(iter_count+1)
                break
            iter_count += 1

        sentiment_dict = run(field_content, sentiment_dict, sentence_train)
        train_feature_solve(field_content, lexcion, sentence_train)
        test_sentences = load_pickle_file(field_content+r"pickles/test_parse_sentences.pickle")
        test_feature_solve(field_content, lexcion, test_sentences)
        train_and_test_solve(field_content)



