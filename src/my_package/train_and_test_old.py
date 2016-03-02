# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
from my_package.scripts import create_content, load_pickle_file, all_match,\
    have_part, all_cover, load_file_line
from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
import sys, getopt
from collections import defaultdict

def train_and_classifiy(feature_vector_content):
    ''''''
    X_all, y_all = load_svmlight_file(feature_vector_content + r"all_match_feature_vectors.txt")
    print("train set len:", len(y_all))
    #  print("random classifier")
    #  clf1= RandomForestClassifier(n_estimators=1000)
    #  clf1 = svm.LinearSVC()
    clf1 = LogisticRegression(C=1.0, intercept_scaling=1, dual=False,
            fit_intercept=True, penalty="l2", tol=0.0001)
    #  clf1 = svm.SVC(kernel='linear', probability=True)
    print("fit..")
    clf1.fit(X_all, y_all)
    print("fit end...")
    '''
    X_all, y_all = load_svmlight_file(feature_vector_content + r"all_cover_feature_vectors.txt")
    clf2= RandomForestClassifier(n_estimators=1000)
    #clf2 = svm.LinearSVC()
    clf2.fit(X_all, y_all)


    X_all, y_all = load_svmlight_file(feature_vector_content + r"have_part_feature_vectors.txt")
    clf3= RandomForestClassifier(n_estimators=1000)
    #clf3 = svm.LinearSVC()
    clf3.fit(X_all, y_all)
    '''

    return clf1, clf1, clf1

def get_sort(t):
    '''根据概率矩阵返回每类经过排序以后的行号
    t -- 概率矩阵

    return

    {类别号：[根据概率排序过后的行号,...], ...}

    '''

    index = np.array([np.arange(t.shape[0])]).transpose()
    max_value = np.array([t.max(axis=1)]).transpose()
    max_index = np.array([t.argmax(axis=1)]).transpose()
    index_max_index_max_value = np.hstack([index, max_index, max_value])

    #df = DataFrame(index_max_index_max_value)
    #df = df.sort_index(by=1, ascending=False)
    d = defaultdict(list)
    #for key, value in df[1].groupby(df[0]).groups.items():
        #d[int(key)] = value
        #[int(line[0]) for line in sorted(df.ix[value][[0, 2]].values, key=lambda x : x[1], reverse=True)]
    for line in index_max_index_max_value:
        d[line[1]].append((line[0], line[2]))
    ret_dict = defaultdict(list)
    for key, value in d.items():
        ret_dict[key] = [int(line[0]) for line in sorted(value, key=lambda x:x[1], reverse=True)]
    return ret_dict[1]

def test_and_classifiy(field_content, feature_vector_path, clf, f):
    X_test, y_test = load_svmlight_file(feature_vector_path)

    sentences = load_pickle_file(field_content + r"pickles/test_feature_vector_sentences.pickle")
    y = clf.predict(X_test)
    i = 0
    all_true = 0
    predict_true = 0
    all_predict_true = 0
    if f == all_match:
        out_path = field_content + r"results/all_match_predict_text.txt"
        res_path = field_content + r"results/all_match_res.txt"
    elif f == all_cover:
        out_path = field_content + r"results/all_cover_predict_text.txt"
        res_path = field_content + r"results/all_cover_res.txt"
    else:
        out_path = field_content + r"results/have_part_predict_text.txt"
        res_path = field_content + r"results/have_part_res.txt"
    fout = open(out_path, "w", encoding="utf8")
    for sentence in sentences:
        print(sentence.text, file=fout)
        all_true += len(sentence.feature_sentiment)
        tuple_pair_set = set()
        predict_true_pair_set = set()
        for pair in sentence.candidate_pairs:

            if y[i] != 0:
                print(sentence.get_phrase(pair[0]), "   ", sentence.get_phrase(pair[1]), y[i], end="", file=fout)
                #计算召回率
                flag = 0
                for fs_pair in list(sentence.fs_dict.keys()):
                    if f(pair, fs_pair):
                        flag = 1
                        tuple_pair_set.add(fs_pair)
                        break
                print("      ", flag, end="  ", file=fout)
                if flag:
                    print("找到关系并且分类正确", file=fout)
                else:
                    print("找到了错误的关系", file=fout)
                #计算精确率
                flag = 1
                for true_pair in predict_true_pair_set:
                    if f(pair, true_pair):
                        flag = 0
                        break
                if flag:
                    predict_true_pair_set.add(pair)
            else:
                flag = 0
                for fs_pair in list(sentence.fs_dict.keys()):
                    if f(pair, fs_pair):
                        flag = 1
                        break
                if flag:
                    print(sentence.get_phrase(pair[0]), "   ", sentence.get_phrase(pair[1]), y[i], end="", file=fout)
                    print("      ", flag, "   找到关系但是分类错误", file=fout)
            i += 1

        #print(tuple_pair_set, list(sentence.fs_dict.keys()))
        for t in sentence.fs_dict.keys():
            print(sentence.get_phrase(t[0]), "   ", sentence.get_phrase(t[1]), end="   ", file=fout)
            if t in tuple_pair_set:
                print("找到的关系", file=fout)
            else:
                print("没有找到的关系", file=fout)
        print("\n", file=fout)

        predict_true += len(tuple_pair_set)
        all_predict_true += len(predict_true_pair_set)
    fout.close()

    f = open(res_path, "w", encoding="utf8")
    print(predict_true, all_predict_true, all_true)
    P = predict_true / all_predict_true
    R = predict_true / all_true
    F = 2 * P * R / (P + R)
    print("P:", P)
    print("R:", R)
    print("F:", F)
    print("\n")

    print(predict_true, all_predict_true, all_true, file=f)
    print("P:", P, file=f)
    print("R:", R, file=f)
    print("F:", F, file=f)
    f.close()


def adjust_train_set(feature_vector_content, r=1):
    for file_name in os.listdir(feature_vector_content):
        if file_name[:3] != "raw":
            continue
        f = open(feature_vector_content+file_name[4:], "w", encoding="utf8")
        P, N = [], []
        for line in load_file_line(feature_vector_content+file_name):
            if line[0] == "0":
                N.append(line)
            else:
                P.append(line)
        random.shuffle(N)
        n = int(r*len(P))
        P.extend(N[:n])
        random.shuffle(P)
        for line in P:
            print(line, file=f)
        f.close()

def validation_and_classifiy(field_content, feature_vector_path, clf, have_part):
    X_test, y_test = load_svmlight_file(feature_vector_path)
    y_prob = clf.predict_proba(X_test)
    y = clf.predict(X_test)
    y_confidence = get_sort(y_prob)

    return y_confidence

def train_and_validation_solve(field_content):
    create_content(field_content + "results")
    adjust_train_set(field_content+"feature_vectors/", 1000)

    clf1, clf2, clf3 = train_and_classifiy(field_content + r"feature_vectors/")
    feature_vector_path = field_content + r"feature_vectors/test_feature_vectors.txt"

    #test_and_classifiy(field_content, feature_vector_path, clf1, all_match)

    #test_and_classifiy(field_content, feature_vector_path, clf1, all_cover)

    return validation_and_classifiy(field_content, feature_vector_path, clf1, have_part)

def train_and_test_solve(field_content, r):
    create_content(field_content + "results")
    adjust_train_set(field_content+"feature_vectors/", r)

    clf1, clf2, clf3 = train_and_classifiy(field_content + r"feature_vectors/")
    feature_vector_path = field_content + r"feature_vectors/test_feature_vectors.txt"

    test_and_classifiy(field_content, feature_vector_path, clf1, all_match)

    test_and_classifiy(field_content, feature_vector_path, clf1, all_cover)

    test_and_classifiy(field_content, feature_vector_path, clf1, have_part)

def usage():
    '''打印帮助信息'''
    print("train_and_test.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-r, --rate： 负样本与正样本数量的比值")


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:r:", ["help", "domain=", "rate="])
    except getopt.GetoptError:
        print("命令行参数输入错误！")
        usage()
        sys.exit(1)
    for op, value in opts:
        if op in ("-h", "--help"):
            usage()
            sys.exit()
        if op in ("-d", "--domain"):
            content = value
        if op in ("-r", "--rate"):
            r = float(value)
    field_content = r"../../data/domains/" + content + r"/"
    train_and_test_solve(field_content, r)

