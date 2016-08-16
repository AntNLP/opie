#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/12 19:40:09

@author: Changzhi Sun
"""
import getopt
import sys
import os

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import mkdir
from my_package.static import Static


class ComplexWordGenerator:

    def __init__(self, domain, pivot_word, mysql_db=None):
        self.domain = domain
        self.domain_dir = os.path.join(
            os.getenv("OPIE_DIR"), "data/domains", domain)
        self.pickle_dir = os.path.join(self.domain_dir, "pickles")
        self.pickle_parse = os.path.join(self.pickle_dir, "parse_sentences")
        self.near_dir = os.path.join(self.domain_dir, "near")
        self.pivot_word = pivot_word
        self.table_opinwd = set(Static.opinwd.keys())
        self.complex_word = {}
        self.mysql_db = mysql_db
        mkdir(self.near_dir)

    def match_complex_condition(self, sentence, phrase):
        if len(phrase) == 1:
            return False
        for i in phrase:
            if sentence.is_weak_opinwd(sentence.tokens[i].lower()):
                return False
        complex_word_str = sentence.print_phrase(phrase).lower()
        if (self.mysql_db and
                not self.mysql_db.language_model(complex_word_str)):
            return False
        return True

    def handle_phrase(self, sentence, pos_tag, i, idx_pivot):
        if sentence.pos_tag[i] in pos_tag:
            if pos_tag == Static.VERB:
                phrase = sentence.get_max_VP(i, idx_pivot)
            else:
                phrase = sentence.get_max_ADJP(i, idx_pivot)
            complex_word_str = sentence.print_phrase(phrase).lower()
            if (self.match_complex_condition(sentence, phrase) and
                    complex_word_str not in self.complex_word):
                self.complex_word[complex_word_str] = " ".join(
                    [sentence.pos_tag[e] for e in phrase])

    def get_complex_word(self, sentence, begin, end, idx_pivot):
        for i in range(begin, end):
            self.handle_phrase(sentence, Static.VERB, i, idx_pivot)
            self.handle_phrase(sentence, Static.ADJ, i, idx_pivot)

    def handle_sentence(self, sentence, w=5):
        n = len(sentence.tokens)
        for i in range(1, n+1):
            j, k = i, 0
            while(k < len(self.pivot_word) and
                    j <= n and sentence.tokens[j] == self.pivot_word[k]):
                j += 1
                k += 1
            if k != len(self.pivot_word):
                continue
            b = i - w if i - w >= 1 else 1
            e = j + w if j + w <= n + 1 else n + 1
            self.get_complex_word(sentence, b, i, i)
            self.get_complex_word(sentence, j, e, j-1)

    def handle_sentences(self, sentences):
        for sentence in sentences:
            self.handle_sentence(sentence)

    def print_complex_word(self):
        with open(os.path.join(self.near_dir, "complex_word_pos_tag"),
                  "w", encoding="utf8") as f:
            for word, pos_tag in self.complex_word.items():
                print("0\t%s\t%s" % (word, pos_tag), file=f)


def usage():
    '''print help information'''
    print("complex_word_generator.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-p, --pivot: pivot word")
    print("-b, --begin: ")
    print("-e, --end: ")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hp:b:e:d:",
            ["help", "pivot=", "begin=", "end=", "domain="])
    except getopt.GetoptError:
        print("命令行参数输入错误！")
        usage()
        sys.exit(1)
    for op, value in opts:
        if op in ("-h", "--help"):
            usage()
            sys.exit()
        if op in ("-d", "--domain"):
            domain = value
        if op in ("-p", "--pivot"):
            pivot_word = value.split(' ')
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)
    c = ComplexWordGenerator(domain, pivot_word)
    i = b
    filename = os.path.join(c.pickle_parse, "parse_sentences_%d.pickle" % i)
    while i < e and os.path.exists(filename+".bz2"):
        print("pickle index: %d loading" % i)
        sentences = load_pickle_file(filename)
        print("pickle index: %d loaded" % i)
        c.handle_sentences(sentences)
        i += 1
        filename = os.path.join(c.pickle_parse,
                                "parse_sentences_%d.pickle" % i)
    c.print_complex_word()
