#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/05 19:08:39

@author: Changzhi Sun
"""
import re
import configparser

import nltk
from nltk.tokenize import TreebankWordTokenizer

from my_package.scripts import save_pickle_file
from my_package.static import get_wpath


class SentenceTokenizer:

    # The constructor builds a classifier using treebank training data
    # Naive Bayes is used for fast training
    # The entire dataset is used for training
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()

        self.word_pattern = re.compile(r"^([\w.]*)(\.)(\w*)$")
        self.proper_noun = re.compile(r"([A-Z]\.){2,}$")

        f = open(get_wpath("transition_words"), "r", encoding="utf8")
        transition_word = f.readline()
        self.words = r"([.,!?;:])\ *" + transition_word
        f.close()

        training_sents = nltk.corpus.treebank_raw.sents()
        tokens = []
        boundaries = set()
        offset = 0
        for sent in training_sents:
            tokens.extend(sent)
            offset += len(sent)
            boundaries.add(offset-1)

        # Create training features
        featuresets = [(self.punct_features(tokens, i), (i in boundaries))
                       for i in range(1, len(tokens)-1)
                       if tokens[i] in '.?!']

        train_set = featuresets
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    # extract punctuation features from word list for position i
    # Features are: this word; previous word (lower case);
    # is the next word capitalized?; previous word only one char long?
    def punct_features(self, tokens, i):
        return {
            'next-word-capitalized': ((i < len(tokens)-1) and
                                      (tokens[i+1][0].isupper())),
            'next-word': ((i < len(tokens)-1) and
                          re.search(r"^\w+$", tokens[i+1])),
            'prevword': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1
            }

    # Use the classifier to segment word tokens into sentences
    # words is a list of (word,bool) tuples
    def classify_segment_sentences(self, words):
        start = 0
        sents = []
        for i, word in enumerate(words):

            if (word in '.?!' and
                    self.classifier.classify(
                        self.punct_features(words, i)) == True):
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            sents.append(words[start:])
        return sents

    def handle_word(self, text):
        text_words_sp = []
        for t in text:
            if re.search(r"\.{3,}", t):
                text_words_sp.append(t)
                continue
            if re.search(r"^[0-9]+\.[0-9]+", t):
                text_words_sp.append(t)
                continue
            if self.proper_noun.search(t):
                text_words_sp.append(t)
                continue
            obj = self.word_pattern.search(t)

            if obj:
                if obj.groups()[0] != "":
                    if obj.groups()[2] != "":
                        text_words_sp.extend(obj.groups())
                    else:
                        text_words_sp.extend(obj.groups()[0:2])
                else:
                    text_words_sp.extend(obj.groups()[-2:])
            else:
                text_words_sp.extend(
                    re.sub(r"\.", r"##########.", t).split("##########"))
        return text_words_sp

    # Segment text into sentences and words
    # returns a list of sentences, each sentence is a list of words
    # punctuation chars are classed as word tokens (except abbreviations)
    def nltk_check(self, sentences):
        sents = []
        for s in sentences:
            t = nltk.sent_tokenize(" ".join(s))
            for tt in t:
                ss = re.sub(r"^[^a-zA-Z(]*", "", tt)
                ss = re.sub(r"\(", "-LRB-", ss)
                ss = re.sub(r"\)", "-RRB-", ss)
                if re.search(r"^\W*$", ss):
                    continue
                sents.append(ss)

        return sents

    def split_junction(self, sentences):
        split_pattern = re.compile(self.words, re.I)
        sents = []
        for sentence in sentences:
            if len(sentence.split(" ")) < 20:
                sents.append(sentence)
                continue
            sent = re.split(r"##########",
                            split_pattern.sub(r"\1##########\2", sentence))
            for w in sent:
                if len(w.split(" ")) >= 20:
                    w_list = re.split(r"\ [,;:]\ ", w)
                    sents.extend(w_list)
                else:
                    sents.append(w)
        return sents

    def segment_text(self, full_text):
        # Split (tokenize) text into words. Count whitespace as
        # words. Keeping this information allows us to distinguish between
        # abbreviations and sentence terminators
        text_words_sp_temp = self.tokenizer.tokenize(full_text)
        text_words_sp = self.handle_word(text_words_sp_temp)
        text_words_sp = [w for w in text_words_sp if w]
        sentences = self.classify_segment_sentences(text_words_sp)
        sentences = self.nltk_check(sentences)
        sentences = self.split_junction(sentences)
        return sentences

if __name__ == "__main__":
    my_tokenizer = SentenceTokenizer()
