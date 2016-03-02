# -*- coding: utf-8 -*-
'''
Created on 2015年10月3日

@author: Changzhi Sun
'''
import os
import shutil
from my_package.scripts import create_content
import sys, getopt

def split_data(input_path):
    '''将所有句子平均分为 8 份，然后并行 parse'''
    n = int(len(os.listdir(input_path)) / 8)
    k = 1
    filename = input_path + "sentences_" + str(k) + ".txt"
    for i in range(1, 9):
        if os.path.exists(input_path + "Part" + str(i)):
            shutil.rmtree(input_path + "Part" + str(i))
        create_content(input_path + "Part" + str(i))
    i = 1
    while os.path.exists(filename):
        shutil.move(filename, input_path + "Part" + str(i))
        if k % n == 0 and i < 8:
            i += 1
        k += 1
        filename = input_path + "sentences_" + str(k) + ".txt"

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
            content = value
    print(content)
    input_path = r"../../data/domains/" + content + r"/sentences/"
    #  create_content(output_path)
    split_data(input_path)
