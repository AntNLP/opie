# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
from my_package.scripts import create_content
import sys, getopt
def usage():
    '''打印帮助信息'''
    print("create_language_model.py 用法:")
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
    input_path = r"../../data/domains/" + content
    create_content(input_path + r"/lm")
    #  cmd_string = r"find " + input_path + r"/sentences -type f | xargs cat > " + input_path + "/text"
    #  os.system(cmd_string)

    cmd_string = r"../../tools/kenlm/bin/lmplz -o 5 <" + input_path + "/text " + ">" + input_path + "/lm/text.arpa"
    os.system(cmd_string)
    #cmd_string = r"rm " + input_path + "/text"
    #os.system(cmd_string)
    print("end")
