#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/09/14 14:59:46

@author: Changzhi Sun
"""
import getopt
import sys
import os
import math
import numpy as np
import re
from itertools import chain

from my_package.sentence import Sentence
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import load_json_file
from my_package.scripts import save_json_file
from my_package.static import Static
import data_helpers

def usage():
    '''print help information'''
    print("review_classification.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hd:",
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data", "domains", domain)
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")
    x_text, y, _ = data_helpers.load_data_and_labels(domain_dir)

