#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/03/14 13:35:52

@author: Changzhi Sun
"""
import getopt
import sys
from itertools import chain

def parse(filename):
    with open(filename, "r", encoding="utf8") as f:
        res = {}
        for line in f:
            strip_line = line.strip()
            e = strip_line.split('\t')
            if e[0] == "S":
                if len(res) != 0:
                    yield res
                    res = {}
                res["S"] = e[1]
            else:
                if "R" not in res:
                    res["R"] = []
                res["R"].append(e[1:])
        yield res

def remove_subset(new):
    res = set()
    for e in new:
        tmp = set()
        m = True
        for r in res:
            if set(r[0]) | set(e[0]) == set(e[0]) and r[1] == e[1]:
                pass
            elif set(r[1]) | set(e[1]) == set(e[1]) and r[0] == e[0]:
                pass
            elif set(r[0]) | set(e[0]) == set(r[0]) and r[1] == e[1]:
                m = False
                tmp.add(r)
            elif set(r[1]) | set(e[1]) == set(r[1]) and r[0] == e[0]:
                m = False
                tmp.add(r)
            else:
                tmp.add(r)
        if m:
            tmp.add(e)
        res = tmp
    return res

def merge_continues(new):
    res = set()
    for e in new:
        tmp = set()
        m = True
        for r in res:
            if e[1] == r[1]:
                t = []
                ch = list(chain(e[0], r[0]))
                min_v, max_v = min(ch), max(ch)
                if max_v - min_v + 1 == len(ch):
                    m = False
                    t.append(tuple(range(min_v, max_v+1)))
                    t.append(e[1])
                    t.append("merge_continues")
                    t = tuple(t)
                    tmp.add(t)
                else:
                    tmp.add(r)
            elif e[0] == r[0]:
                t = []
                ch = list(chain(e[1], r[1]))
                min_v, max_v = min(ch), max(ch)
                if max_v - min_v + 1 == len(ch):
                    m = False
                    t.append(e[0])
                    t.append(tuple(range(min_v, max_v+1)))
                    t.append("merge_continues")
                    t = tuple(t)
                    tmp.add(t)
                else:
                    tmp.add(r)
            else:
                tmp.add(r)
        if m:
            tmp.add(e)
        res = tmp
    return res



def usage():
    '''打印帮助信息'''
    print("normalization.py 用法:")
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

    field_content = r"../../data/domains/" + content + r"/"
    #  field_content = r"../../data/soft_domains/" + content + r"/"
    #  files = [field_content+"bootstrap/bootstarp_general_relation",
             #  field_content+"result/all_match_predict_text"]
    article = set(["the", "a", "an", "'s'", "is", "am", "are", "was", "were", "been", "being", "be"])
    files = [field_content+"bootstrap/bootstrap_test_general_relation"]
    for each_file in files:
        out = open(each_file+".normalize", "w", encoding="utf8")
        for e in parse(each_file):
            print("S\t%s"%e["S"], file=out)
            j, tokens = 1, {}
            for token in e["S"].lower().split(' '):
                tokens[j] = token
                j += 1
            new = set()
            for r in e["R"]:
                feat_pos = [int(e) for e in r[2][1:-1].split(', ')]
                sent_pos = [int(e) for e in r[3][1:-1].split(', ')]
                feat_sent = []
                i = 0
                while i < len(feat_pos) and tokens[feat_pos[i]] in article:
                    i += 1
                feat_sent.append(tuple(feat_pos[i:]))
                i = 0
                while i < len(sent_pos) and tokens[sent_pos[i]] in article:
                    i += 1
                feat_sent.append(tuple(sent_pos[i:]))
                feat_sent.append(r[4])
                new.add(tuple(feat_sent))
            new = remove_subset(new)
            new = merge_continues(new)
            for feat, sent, regu in new:
                print("R\t%s"%(" ".join([tokens[e] for e in feat])), end="\t", file=out)
                print("%s"%(" ".join([tokens[e] for e in sent])), end="\t", file=out)
                print(list(feat), end="\t", file=out)
                print(list(sent), end="\t", file=out)
                print(regu, file=out)
        out.close()

