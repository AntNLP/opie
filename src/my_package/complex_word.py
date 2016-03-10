# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
from my_package.scripts import load_pickle_file, return_none, save_pickle_file, save_json_file, create_content
import os
from collections import Counter
from itertools import chain
from my_package.class_define import Static
import re
import sys, getopt
import pymysql
import math
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import precision_recall_curve
import numpy as np
#  import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

def max_count(count):
    max_i = None
    for key, value in count.items():
        if max_i == None or value > count[max_i]:
            max_i = key
    return max_i


def parse(filename):
    with open(filename, "r", encoding="utf8") as f:
        record = []
        for line in f:
            tmp = line.strip()
            if line.strip() == "":
                yield record
                record = []
            else:
                record.append(tmp)

def extract(near, sentiments):
    X, y = [], []
    mark = []
    words = []
    i = 0
    lexcion = {}
    lex_pos = {}
    pos_tag = []
    f = open(near + "feature_vector_tmp", "w", encoding="utf8")
    for e in parse(near + "complex_word_feature"):
        label, count, pmi_pos, pmi_neg, emi_pos, emi_neg, emi, nwp, word, pos = e[0].split('\t')
        #  if int(count) <= 5:
            #  #  print(word)
            #  continue
        m = 0
        word_index = []
        pos_index = []
        for w in word.split(' '):
            if w not in lexcion:
                lexcion[w] = len(lexcion) + 1
            word_index.append(lexcion[w])
            if w in sentiments:
                m = 1
        for w in pos.split(' '):
            if w not in lex_pos:
                lex_pos[w] = len(lex_pos) + 1
            pos_index.append(lex_pos[w])
        pos_tag.append(set(pos_index))
        mark.append(m)
        words.append(word)

        y.append(int(label))
        print("%s"%label, end="", file=f)
        print(" 1:%d"%len(word.split(' ')), end="", file=f)
        print(" 2:%d"%int(count), end="", file=f)
        print(" 3:%f"%float(pmi_pos), end="", file=f)
        print(" 4:%f"%float(pmi_neg), end="", file=f)
        print(" 5:%f"%float(emi_pos), end="", file=f)
        print(" 6:%f"%float(emi_neg), end="", file=f)
        print(" 7:%f"%float(emi), end="", file=f)
        #  print(" 8:%f"%float(nwp), end="", file=f)
        #  sum_pre, sum_cur, sum_nex = 0, 0, 0
        #  sum_min_pre_cur, sum_min_pre_nex, sum_min_cur_nex = 0, 0, 0
        #  sum_max_pre_cur, sum_max_pre_nex, sum_max_cur_nex = 0, 0, 0
        #  sum_min_pre_cur_nex = 0
        #  sum_max_pre_cur_nex = 0
        #  sum_bool_pre, sum_bool_cur, sum_bool_nex = 0, 0, 0
        #  sum_bool_pre_cur, sum_bool_pre_nex, sum_bool_cur_nex = 0, 0, 0
        #  sum_bool_pre_cur_nex = 0
        first = True
        count_pre, count_cur, count_nex = Counter(), Counter(), Counter()
        for v in e[1:]:
            pre, cur, nex = v.split(' ')
            pre, cur, nex = int(pre), int(cur), int(nex)
            count_pre[pre] += 1
            count_cur[cur] += 1
            count_nex[nex] += 1
            if first:
                first = False
                matrix = np.array([[pre, cur, nex]])
            else:
                matrix = np.vstack((matrix, [pre, cur, nex]))
            #  sum_pre += pre
            #  sum_cur += cur
            #  sum_nex += nex
            #  sum_min_pre_cur += min(pre, cur)
            #  sum_min_pre_nex += min(pre, nex)
            #  sum_min_cur_nex += min(cur, nex)
            #  sum_max_pre_cur += max(pre, cur)
            #  sum_max_pre_nex += max(pre, nex)
            #  sum_max_cur_nex += max(cur, nex)
            #  sum_min_pre_cur_nex += min(pre, cur, nex)
            #  sum_max_pre_cur_nex += max(pre, cur, nex)
            #  sum_bool_pre += int(pre != 0)
            #  sum_bool_cur += int(cur != 0)
            #  sum_bool_nex += int(nex != 0)
            #  sum_bool_pre_cur += int(pre != 0 and cur != 0)
            #  sum_bool_pre_nex += int(pre != 0 and nex != 0)
            #  sum_bool_cur_nex += int(cur != 0 and nex != 0)
            #  sum_bool_pre_cur_nex += int(pre != 0 and cur != 0 and nex != 0)
        max_count_pre, max_count_cur, max_count_nex = max_count(count_pre), max_count(count_cur), max_count(count_nex)
        n = len(e) - 1
        #  print(" 9:%d"%sum_pre, end="", file=f)
        #  print(" 10:%d"%sum_cur, end="", file=f)
        #  print(" 11:%d"%sum_nex, end="", file=f)


        #  print(" 12:%d"%sum_min_pre_cur, end="", file=f)
        #  print(" 13:%d"%sum_min_pre_nex, end="", file=f)
        #  print(" 14:%d"%sum_min_cur_nex, end="", file=f)
        #  print(" 15:%d"%sum_max_pre_cur, end="", file=f)
        #  print(" 16:%d"%sum_max_pre_nex, end="", file=f)
        #  print(" 17:%d"%sum_max_cur_nex, end="", file=f)


        #  print(" 18:%d"%sum_min_pre_cur_nex, end="", file=f)
        #  print(" 19:%d"%sum_max_pre_cur_nex, end="", file=f)


        #  print(" 20:%d"%sum_bool_pre, end="", file=f)
        #  print(" 21:%d"%sum_bool_cur, end="", file=f)
        #  print(" 22:%d"%sum_bool_nex, end="", file=f)


        #  print(" 23:%d"%sum_bool_pre_cur, end="", file=f)
        #  print(" 24:%d"%sum_bool_pre_nex, end="", file=f)
        #  print(" 25:%d"%sum_bool_cur_nex, end="", file=f)


        #  print(" 26:%d"%sum_bool_pre_cur_nex, end="", file=f)



        #  print(" 27:%f"%(sum_pre/n), end="", file=f)
        #  print(" 28:%f"%(sum_cur/n), end="", file=f)
        #  print(" 29:%f"%(sum_nex/n), end="", file=f)


        #  print(" 30:%f"%(sum_min_pre_cur/n), end="", file=f)
        #  print(" 31:%f"%(sum_min_pre_nex/n), end="", file=f)
        #  print(" 32:%f"%(sum_min_cur_nex/n), end="", file=f)
        #  print(" 33:%f"%(sum_max_pre_cur/n), end="", file=f)
        #  print(" 34:%f"%(sum_max_pre_nex/n), end="", file=f)
        #  print(" 35:%f"%(sum_max_cur_nex/n), end="", file=f)


        #  print(" 36:%f"%(sum_min_pre_cur_nex/n), end="", file=f)
        #  print(" 37:%f"%(sum_max_pre_cur_nex/n), end="", file=f)


        #  print(" 38:%f"%(sum_bool_pre/n), end="", file=f)
        #  print(" 39:%f"%(sum_bool_cur/n), end="", file=f)
        #  print(" 40:%f"%(sum_bool_nex/n), end="", file=f)


        #  print(" 41:%f"%(sum_bool_pre_cur/n), end="", file=f)
        #  print(" 42:%f"%(sum_bool_pre_nex/n), end="", file=f)
        #  print(" 43:%f"%(sum_bool_cur_nex/n), end="", file=f)


        #  print(" 44:%f"%(sum_bool_pre_cur_nex/n), end="", file=f)

        #  print(" 9:%d 10:%d 11:%d"%tuple(matrix.max(axis=0)), end="", file=f)
        #  print(" 12:%f 13:%f 14:%f"%tuple(matrix.std(axis=0)), end="", file=f)
        #  print(" 15:%f 16:%f 17:%f"%tuple(matrix.var(axis=0)), end="", file=f)
        #  print(" 18:%f 19:%f 20:%f"%tuple(matrix.mean(axis=0)), end="", file=f)
        #  print(" 21:%f 22:%f 23:%f"%tuple(matrix.min(axis=0)), end="", file=f)
        #  print(" 24:%d 25:%d 26:%d"%(max_count_pre, max_count_cur, max_count_nex), end="", file=f)
        print(" 8:%d 9:%d 10:%d"%(max_count_pre, max_count_cur, max_count_nex), end="", file=f)
        for x in sorted(set(word_index)):
            print(" %d:1"%(x+10), end="", file=f)
        print(file=f)
    f.close()
    f = open(near + "feature_vector_tmp", "r", encoding="utf8")
    g = open(near + "feature_vector", "w", encoding="utf8")
    i = 0
    for line in f:
        print("%s"%(line.strip()), end="", file=g)
        for x in sorted(pos_tag[i]):
            print(" %d:1"%(x+10+len(lexcion)), end="", file=g)
        print(file=g)
        i += 1
    f.close()
    g.close()
    return np.array(words), np.array(mark)

def fit(X_all, y_all):
    clf = LogisticRegression(C=1.0, intercept_scaling=1, dual=False,
            fit_intercept=True, penalty="l2", tol=0.0001)
    print("fit..")
    clf.fit(X_all, y_all)
    print("fit end")
    return clf;

def test_only(clf, X_test, y_test, mark_test, test_words):

    y = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    y_true = np.array(y_test)

    y_test = [y_test[e] for e in range(len(y_test)) if mark_test[e] == 0]
    y = [y[e] for e in range(len(y)) if mark_test[e] == 0]
    y_score = np.array([y_score[e] for e in range(len(y_score)) if mark_test[e] == 0])
    test_words = [test_words[e] for e in range(len(test_words)) if mark_test[e] == 0]
    y_true = [y_true[e] for e in range(len(y_true)) if mark_test[e] == 0]

    k = 0
    for i in range(len(y)):
        if y[i] == 1 and y_test[i] == 1:
            k += 1
    P = k / sum(y)
    R = k / sum(y_test)
    F = 2*P*R/(P+R)
    target_names = ["class 0", "class 1"]
    print("test only")
    print("P", P, k, sum(y))
    print("R", R, k, sum(y_test))
    print("F", F)
    print("f1 score", f1_score(y_test, y))
    print(classification_report(y_test, y, target_names=target_names))
    print(confusion_matrix(y_test, y))

    f = open("true_score.res", "w", encoding="utf8")
    for i in range(len(y_true)):
        print("%s\t%d\t%f\t%d"%(test_words[i], y_true[i], y_score[i, 1], int(y_score[i, 1]>=0.5)), file=f)
    f.close()

    precision, recall, threshold = precision_recall_curve(y_true, y_score[:, 1])
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("only PR curve")
    plt.show()

def test_all(clf, X_test, y_test, mark_test, test_words):

    y = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    y_true = np.array(y_test)

    k = 0
    for i in range(len(y)):
        if y[i] == 1 and y_test[i] == 1:
            k += 1
    P = k / sum(y)
    R = k / sum(y_test)
    F = 2*P*R/(P+R)
    target_names = ["class 0", "class 1"]
    print("test all")
    print("P", P, k, sum(y))
    print("R", R, k, sum(y_test))
    print("F", F)
    print("f1 score", f1_score(y_test, y))
    print(classification_report(y_test, y, target_names=target_names))
    print(confusion_matrix(y_test, y))

    precision, recall, threshold = precision_recall_curve(y_true, y_score[:, 1])
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("all PR curve")
    plt.show()

def calcu_count(connection, sentiment_dict, table_lm, table_posting):
    word_count_pos, word_count_neg = [], []
    print(len(sentiment_dict))
    i = 0
    for key, value in sentiment_dict.items():
        if i % 100 == 0:
            print(i)
        i += 1
        word_index = get_index(connection, table_lm, key)
        if word_index == None:
            print("not index", key)
            continue
        ret = get_positon(connection, table_posting, word_index)
        if ret == None:
            print("posting none", key)
            continue
        res_set = set(((e['i_pickle'], e['i_sentence']) for e in ret))
        if value == 1:
            word_count_pos.append(res_set)
        else:
            word_count_neg.append(res_set)

    return word_count_pos, word_count_neg


def inquire_num(connection, i_pickle, i_sentence, table_num):
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "select num, i_review from %s where i_pickle=%d and i_sentence=%d"%(table_num, i_pickle, i_sentence)
            cursor.execute(sql)
            res = cursor.fetchall()
            return res
    except Exception as err:
        print(err)
        return None
    finally:
        pass

def inquire_total(connection, table_posting):
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "select count(distinct i_pickle, i_sentence) from %s"%table_posting
            cursor.execute(sql)
            res = cursor.fetchone()
            return res
    except Exception as err:
        print(err)
        return None
    finally:
        pass

def calcu_PMI(word_list, word_set, n):
    s = 0
    c = len(word_set) / n
    for each_word in word_list:
        p = len(each_word & word_set)
        if p != 0:
            s += p * math.log(n * p / (c * len(each_word)), 2) / n
        p = len(word_set - each_word)
        if p != 0:
            s += p * math.log(n * p / (c * (n - len(each_word))), 2) / n
        p = len(each_word - word_set)
        if p != 0:
            s += p * math.log(n * p / ((n - c) * len(each_word)), 2) / n
        p = n - len(each_word | word_set)
        if p != 0:
            s += p * math.log(n * p / ((n-c) * (n - len(each_word))), 2) / n
    return s/len(word_list)

def calcu_EMI(word_list, word_set, n):
    s = 0
    for each_word in word_list:
        p = len(each_word & word_set)
        tmp = (len(each_word) - p) * (len(word_set) - p)
        if p != 0 and tmp != 0:
            s += math.log(n * p / tmp, 2)
    return s/len(word_list)

def calcu_emi_nwp(connection, table_lm, res_set, word_string, n, word_count):
    t = n
    c = len(res_set)
    nwp = 1
    for word in word_string.split(' '):
        if word not in word_count:
            word_index = get_index(connection, table_lm, word)
            if word_index == None:
                continue
            res = get_positon(connection, table_posting, word_index)
            word_count[word] = len(res)
        wc = word_count[word]
        if wc > c:
            nwp *= c/(wc-c)
            t *= (wc-c) / n
    return math.log(c/t, 2), nwp

def get_index(connection, table_lm, var):
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "select * from {0} where content=\"{1}\"".format(table_lm, var)
            cursor.execute(sql)
            res = cursor.fetchall()
            if len(res) == 0:
                return None
            else:
                return res[0]['id']

    except Exception as err:
        print(err)
        print(var)
        return None
    finally:
        pass

def get_positon(connection, table_posting, var):
    try:

        # 游标
        with connection.cursor() as cursor:
            sql = "select distinct i_pickle, i_sentence from {0} where i_content={1}".format(table_posting, var)
            cursor.execute(sql)
            res = cursor.fetchall()
            return res
    except Exception as err:
        print(err)
        return None
    finally:
        pass

def run(field_content, word_lists, ret_dict):
    i = 1
    filename = field_content + "pickles/without_parse_sentences/without_parse_sentences_" + str(i) + ".pickle"
    while os.path.exists(filename+".bz2"):
        sentences = load_pickle_file(filename)
        print(filename)
        for j in range(len(sentences)):
            for word in word_lists:
                if judge_sentiment_word(sentences[j], word):
                    key = " ".join(word)
                    if ret_dict.get(key) == None:
                        ret_dict[key] = []
                    cur_num = sentences[j].sentiment_count
                    pre_num = 0 if j == 0 else sentences[j-1].sentiment_count
                    next_num = 0 if j == len(sentences) - 1 else sentences[j+1].sentiment_count
                    ret_dict[key].append([pre_num, cur_num, next_num])
        i += 1
        filename = field_content + "pickles/without_parse_sentences/without_parse_sentences_" + str(i) + ".pickle"

def load_near_word(filename):
    with open(filename, "r", encoding="utf8") as out:
        for line in out:
            yield line.strip().split('\t')

def create_feature(connection, field_content, table_lm, table_posting, table_num):
    if not os.path.exists(field_content+"pickles/word_count_pos.pickle.bz2"):
        word_count_pos, word_count_neg = calcu_count(connection, Static.sentiment_word, table_lm, table_posting)
        save_pickle_file(field_content+"pickles/word_count_pos.pickle", word_count_pos)
        save_pickle_file(field_content+"pickles/word_count_neg.pickle", word_count_neg)
    else:
        word_count_pos = load_pickle_file(field_content+"pickles/word_count_pos.pickle")
        word_count_neg = load_pickle_file(field_content+"pickles/word_count_neg.pickle")

    word_count_pos = list(sorted(word_count_pos, key=lambda x : len(x), reverse=True))
    word_count_neg = list(sorted(word_count_neg, key=lambda x : len(x), reverse=True))
    word_count_pos = word_count_pos[:100]
    word_count_neg = word_count_neg[:100]
    word_count = {}
    if not os.path.exists(field_content+"pickles/pickle_sentence_count.pickle.bz2"):
        ret = inquire_total(connection, table_posting)
        for key, value in ret.items():
            n = value
        save_pickle_file(field_content+"pickles/pickle_sentence_count.pickle", n)
    else:
        n = load_pickle_file(field_content+"pickles/pickle_sentence_count.pickle")
    print("n=", n)
    word_pickle_sentence = load_pickle_file(field_content+"pickles/word_pickle_sentence.pickle")
    f = open(field_content+"near/complex_word_feature", "w", encoding="utf8")
    for word_label, word_string, word_pos in load_near_word(field_content+"near/complex_word_label_pos"):
        res_set = word_pickle_sentence[word_string]
        pos_pmi = calcu_PMI(word_count_pos, res_set, n)
        neg_pmi = calcu_PMI(word_count_neg, res_set, n)
        pos_emi = calcu_EMI(word_count_pos, res_set, n)
        neg_emi = calcu_EMI(word_count_neg, res_set, n)
        emi, nwp = calcu_emi_nwp(connection, table_lm, res_set, word_string, n, word_count)
        print("%d\t%d\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%s\t%s"%(int(word_label), len(res_set), pos_pmi, neg_pmi,
            pos_emi, neg_emi, emi, nwp, word_string, word_pos), file=f)
        for i_pickle, i_sentence in res_set:
            pre_num, cur_num, next_num = 0, 0, 0
            res1 = inquire_num(connection, i_pickle, i_sentence-1, table_num)
            res2 = inquire_num(connection, i_pickle, i_sentence, table_num)
            res3 = inquire_num(connection, i_pickle, i_sentence+1, table_num)
            cur_num = res2[0]['num']
            if len(res1) == 1 and res1[0]['i_review'] == res2[0]['i_review']:
                pre_num = res1[0]['num']
            if len(res3) == 1 and res3[0]['i_review'] == res2[0]['i_review']:
                next_num = res3[0]['num']
            print(pre_num, cur_num, next_num, file=f)
        print(file=f)
    f.close()

def my_cross_validation(X, y, mark, words, cv=10):
    kf = KFold(len(y), n_folds=cv)
    scores, scores_without = [], []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mark_train, mark_test = mark[train_index], mark[test_index]
        words_train, words_test = words[train_index], words[test_index]
        clf = LogisticRegression(C=1.0, intercept_scaling=1, dual=False,
                fit_intercept=True, penalty="l2", tol=0.0001)
        clf.fit(X_train, y_train)
        y_predict = np.sign(clf.predict(X_test))
        y_predict = [int(e) for e in y_predict]
        scores.append(f1_score(y_test, y_predict))
        y_test = [y_test[e] for e in range(len(mark_test)) if mark_test[e] == 0]
        y_predict= [y_predict[e] for e in range(len(mark_test)) if mark_test[e] == 0]
        scores_without.append(f1_score(y_test, y_predict))
    return np.array(scores), np.array(scores_without)

def usage():

    '''打印帮助信息'''
    print("complex_word.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:", ["help", "domain="])
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
    field_content = r"../../data/soft_domains/" + content + r"/"
    create_content(field_content + "near")
    table_lm = content+"_lm"
    table_posting = content + "_posting"
    table_num = content + "_num"

    connection = pymysql.connect(host="localhost",
                                user="u20130099",
                                passwd="u20130099",
                                local_infile=True,
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    create_feature(connection, field_content, table_lm, table_posting, table_num)

    sentiments = set(Static.sentiment_word.keys())
    words, mark = extract(field_content+"near/", sentiments)
    X, y = load_svmlight_file(field_content+"near/feature_vector")
    X = X.toarray()

    scores, socres_without = my_cross_validation(X, y, mark, words)
    print(scores)
    print(scores.mean())

    print(scores_without)
    print(scores_without.mean())
    connection.close()
    print("end")
