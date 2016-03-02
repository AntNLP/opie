# -*- coding: utf-8 -*-
'''
Created on 2016年1月1日

@author: Changzhi Sun
'''
import sys, getopt
import os
from my_package.class_define import Sentence
from my_package.scripts import load_pickle_file, save_pickle_file, save_json_file, load_json_file
import numpy as np

def usage():
    '''打印帮助信息'''
    print("merge_sentiment_pos_num.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:", ["help", "domain"])
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
    print(content)
    #  field_content = r"../../data/soft_domains/" + content + r"/"
    field_content = r"../../data/domains/" + content + r"/"
    #  f = open(field_content + "/pickles/posting", "w", encoding="utf8")
    g = open(field_content+"/pickles/seed_sent_num", "w", encoding="utf8")
    for i in range(1, 9):
        print(i)
        #  with open(field_content + "pickles/posting_{0}.txt".format(i), "r", encoding="utf8") as out:
            #  for line in out:
                #  print(line, end="", file=f)

        with open(field_content + "pickles/seed_sent_num_{0}".format(i), "r", encoding="utf8") as out:
            for line in out:
                print(line, end="", file=g)
    #  f.clsose()
    g.close()
