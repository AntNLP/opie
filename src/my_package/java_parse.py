# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
import shutil
from my_package.scripts import create_content
import sys, getopt
import cmd

def parse_field(input_path, output_path, content, part_count):
    ''''''
    old_pwd = os.getcwd()
    pwd_path = r"../../tools/stanford-corenlp-full-2014-08-27"
    os.chdir(pwd_path)
    cmd_string = r"java -cp "
    cmd_string += r"stanford-corenlp-3.4.1.jar:stanford-corenlp-3.4.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -ssplit.eolonly true -tokenize.whitespace true -annotators tokenize,ssplit,pos,parse -file "
    cmd_string += input_path + "Part" + str(part_count)
    cmd_string += r" -outputDirectory " + output_path + " -noClobber"
    cmd_string += r" 2>&1 | tee ../" + content + "_Part" + str(part_count) + r".log"
    os.system(cmd_string)
    os.chdir(old_pwd)

def usage():
    '''打印帮助信息'''
    print("java_parse.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-p, --part: 每个领域句子均分为 8 份, 该参数指定 parse 哪部分(1,2, ..., 8)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:p:", ["help", "domain=", "part="])
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
        if op in ("-p", "--part"):
            part_count = value
    if os.path.exists(r"../../data/soft_domains/" + content):
        input_path = r"../../data/domains/" + content + r"/sentences/"
        output_path = r"../../data/soft_domains/" + content + r"/parse"
        cmd_string = r"find " + output_path + " -size 0 | xargs rm -f"
        os.system(cmd_string) 
        create_content(output_path)
        parse_field(input_path, output_path, content, part_count)
    else:
        print("路径名称不正确！！！")
