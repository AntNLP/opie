# -*- coding: utf-8 -*-
'''
Created on 2015年10月3日

@author: Changzhi Sun
'''
import os
import shutil
import sys
import getopt

from my_package.scripts import mkdir


def split_data(input_path):
    '''将所有句子平均分为 8 份，然后并行 parse'''

    n = int(len(os.listdir(input_path)) / 8)
    k = 1
    filename = os.path.join(input_path, "sentences_%d.txt" % k)
    for i in range(1, 9):
        if os.path.exists(os.path.join(input_path, "Part%d" % i)):
            shutil.rmtree(os.path.join(input_path, "Part%d" % i))
        mkdir(os.path.join(input_path, "Part%d" % i))
    i = 1
    while os.path.exists(filename):
        shutil.move(filename, os.path.join(input_path, "Part%d" % i))
        if k % n == 0 and i < 8:
            i += 1
        k += 1
        filename = os.path.join(input_path, "sentences_%d.txt" % k)


def usage():
    '''打印帮助信息'''
    print("split_data.py 用法:")
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
            domain = value
    input_path = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain, "sentences")
    split_data(input_path)
