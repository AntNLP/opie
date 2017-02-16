#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/05/04 18:57:00

@author: Changzhi Sun
"""
import os
import sys
import getopt
import configparser

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from pattern_extractor import PatternExtractor
from relation_classifier import RelationClassifier
from pattern_extractor import write
from normalization import handle_normalize
from normalization import parse
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file


def have_overlap(e1, e2):
    if set(e1[0]) & set(e2[0]) != set() and set(e1[1]) & set(e2[1]) != set():
        return True
    return False

def convert_index(sentences, begin, end):
    begin, end = int(begin), int(end)
    for i, sentence in enumerate(sentences):
        if len(sentence) < end:
            begin -= len(sentence)
            end -= len(sentence)
        else:
            tokens = sentence.strip().split(' ')
            j = 1
            base = 0
            while base != begin:
                base += len(tokens[j-1]) + 1
                j += 1
            first = j
            while base + len(tokens[j-1]) != end:
                base += len(tokens[j-1]) + 1
                j += 1
            end = j
            return i, tuple(range(first, end+1))

def handle_review(review_name, ann_name, sent_ann):
    f = open(review_name, "r", encoding="utf8")
    g = open(ann_name, "r", encoding="utf8")
    sentences = f.readlines()
    sents = [e.strip() for e in sentences]
    review = "".join(sentences)
    ann_dict = {}
    #  print(sentences)
    for line in g:
        line_strip = line.strip()
        #  print(line_strip)
        if line_strip.startswith("T"):
            T, O, t = line_strip.split('\t')
            offset = O.split(' ')
            ann_dict[T] = convert_index(sentences, offset[1], offset[2])
        elif line_strip.startswith("R"):
            R, s = line_strip.split('\t')
            arg1, arg2 = s[17:].split(' ')
            arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]
            opwd = ann_dict[arg1]
            optg = ann_dict[arg2]
            if sents[opwd[0]] not in sent_ann:
                sent_ann[sents[opwd[0]]] = set()
            sent_ann[sents[opwd[0]]].add((optg[1], opwd[1]))
    f.close()
    g.close()


def get_ann(ann_dir):
    i = 1
    review_name = os.path.join(ann_dir, "review_%d.txt" % i)
    ann_name = os.path.join(ann_dir, "review_%d.ann" % i)
    sent_ann = {}
    while os.path.exists(review_name):
        handle_review(review_name, ann_name, sent_ann)
        i += 1
        review_name = os.path.join(ann_dir, "review_%d.txt" % i)
        ann_name = os.path.join(ann_dir, "review_%d.ann" % i)
    return sent_ann


def calcu_PRF(filename, sent_ann):
    TP_FP = 0
    TP = 0
    TP_TN = 0
    for e in parse(filename):
        text = e['S']
        TP_FP += len(e['R'])
        unique_set = set()
        if text in sent_ann:
            for rr in e['R']:
                r = (tuple(eval(rr[2])), tuple(eval(rr[3])))
                for ee in sent_ann[text]:
                    if have_overlap(r, ee):
                        unique_set.add(ee)
        TP += len(unique_set)
    for key, value in sent_ann.items():
        TP_TN += len(value)
    P = TP / TP_FP
    R = TP / TP_TN
    F = 2 * P * R / (P + R)
    print("TP:", TP)
    print("TP_FP:", TP_FP)
    print("TP_TF:", TP_TN)
    print("P:", P)
    print("R:", R)
    print("F:", F)


def train_and_test(domain_dir, sentences):
    train_dir = os.path.join(domain_dir, "train")
    test_dir = os.path.join(domain_dir, "test")
    X_train, y_train = load_svmlight_file(os.path.join(train_dir, "feature_vector"))
    X_test, y_test = load_svmlight_file(os.path.join(test_dir, "feature_vector"))
    clf = LogisticRegression(C=1.0, intercept_scaling=1, dual=False,
                             fit_intercept=True, penalty="l2", tol=0.0001)
    print("fit..")
    clf.fit(X_train, y_train)
    print("fit end...")
    y_train_predict = clf.predict(X_train)
    print(f1_score(y_train, y_train_predict))
    y = clf.predict(X_test)
    f = open(os.path.join(test_dir, "relation.classifier"), "w", encoding="utf8")
    i = 0
    for sentence in sentences:
        flag = False
        str_list = []
        str_list.append("S\t{0}".format(sentence.text))
        for pair in sentence.candidate_relation:
            if y[i] != 0:
                flag = True
                str_list.append("R\t{0}\t{1}\t{2}\t{3}".format(
                    sentence.print_phrase(pair[0]).lower(),
                    sentence.print_phrase(pair[1]).lower(),
                    list(pair[0]),
                    list(pair[1])))
            i += 1
        if flag:
            for s in str_list:
                print(s, file=f)
    f.close()


def combine_result(f1, f2, outfile):
    out = open(outfile, "w", encoding="utf8")
    ann = {}
    for e in parse(f2):
        if e["S"] not in ann:
            ann[e["S"]] = e["R"]
        else:
            ann[e["S"]].extend(e["R"])
    for e in parse(f1):
        if e["S"] not in ann:
            ann[e["S"]] = e["R"]
        else:
            ann[e["S"]].extend(e["R"])
    for key, value in ann.items():
        print("S\t%s"%key, file=out)
        t = []
        m = set()
        for v in value:
            if (v[1], v[2]) not in m:
                m.add((v[1], v[2]))
                t.append(v)
        for v in t:
            print("R\t%s"%("\t".join(v)), file=out)
    out.close()


def usage():
    '''print help information'''
    print("test_relation.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:",
                                   ["help", "domain="])
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
    r = PatternExtractor(domain)
    c = RelationClassifier(domain)
    test_dir = os.path.join(r.domain_dir, "test")
    ann_dir = os.path.join(test_dir, "ann")
    sent_ann = get_ann(ann_dir)
    sentences = load_pickle_file(os.path.join(test_dir, "sentences.pickle"))
    r.handle_sentences(sentences)
    with open(os.path.join(test_dir, "relation.pattern"), "w", encoding="utf8") as f:
        write(sentences, f)
    handle_normalize(os.path.join(test_dir, 'relation.pattern'))

    c = load_pickle_file(os.path.join(c.pickle_dir, "relation_classifier.pickle"))
    c.run_test(sentences)
    train_and_test(r.domain_dir, sentences)
    handle_normalize(os.path.join(test_dir, 'relation.classifier'))
    combine_result(os.path.join(test_dir, 'relation.pattern.normalize'),
                   os.path.join(test_dir, 'relation.classifier.normalize'),
                   os.path.join(test_dir, 'relation.combine'))
    handle_normalize(os.path.join(test_dir, 'relation.combine'))
    print("\npattern result")
    calcu_PRF(os.path.join(test_dir, 'relation.pattern.normalize'), sent_ann)
    print("\nclassifier result")
    calcu_PRF(os.path.join(test_dir, 'relation.classifier.normalize'), sent_ann)
    print("\nconbine result")
    calcu_PRF(os.path.join(test_dir, 'relation.combine.normalize'), sent_ann)

    #  i = 1
    #  review_name = os.path.join(ann_dir, "review_%d.txt" % i)
    #  f = open("s", "w", encoding="utf8")
    #  while os.path.exists(review_name):
        #  g = open(review_name, "r", encoding="utf8")
        #  for line in g:
            #  line_strip = line.strip()
            #  tokens = line_strip.split(' ')
            #  if line_strip in sent_ann:
                #  print("S\t%s" % line_strip, file=f)
                #  for e in sent_ann[line_strip]:
                    #  print("R\t{0}\t{1}\t{2}\t{3}".format(
                        #  " ".join([tokens[k-1] for k in e[0]]),
                        #  " ".join([tokens[k-1] for k in e[1]]),
                        #  list(e[0]), list(e[1])), file=f)
        #  g.close()
        #  i += 1
        #  review_name = os.path.join(ann_dir, "review_%d.txt" % i)
    #  f.close()
