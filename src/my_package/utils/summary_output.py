#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/09/22 16:18:08

@author: Changzhi Sun
"""
from process_raw_data import parse
from my_package.scripts import mkdir


def usage():
    '''print help information'''
    print("summary_output.py :")
    print("-h, --help: print help information")
    print("-d, --domain: domain name")

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
            domain = value
    raw_dpath = os.path.join(os.getenv("OPIE_DIR"), "data/raw/domains")
    domain_path = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
    fname = domain + ".json.gz"
    mkdir(domain_path)
