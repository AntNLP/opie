# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
import shutil
import sys
import getopt
import cmd

from my_package.scripts import mkdir


def parse_sentence(input_path, output_path, content, part_count):
    old_pwd = os.getcwd()
    pwd_path = os.path.join(os.getenv("OPIE_DIR"),
                            "tools/stanford-corenlp-full-2014-08-27")
    os.chdir(pwd_path)
    cmd_string = "java -cp "
    cmd_string += "stanford-corenlp-3.4.1.jar"
    cmd_string += ":stanford-corenlp-3.4.1-models.jar"
    cmd_string += ":xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g "
    cmd_string += "edu.stanford.nlp.pipeline.StanfordCoreNLP "
    cmd_string += "-ssplit.eolonly true "
    cmd_string += "-tokenize.whitespace true "
    cmd_string += "-annotators tokenize,ssplit,pos,parse -file "
    cmd_string += input_path + "/Part" + str(part_count)
    cmd_string += " -outputDirectory " + output_path + " -noClobber"
    cmd_string += " 2>&1 | tee ../" + content + "_Part"
    cmd_string += str(part_count) + ".log"
    os.system(cmd_string)
    os.chdir(old_pwd)


def usage():
    '''打印帮助信息'''
    print("java_parse.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-p, --part: 每个领域句子均分为 8 份,该参数指定 parse 哪部分")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:p:",
                                   ["help", "domain=", "part="])
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
        if op in ("-p", "--part"):
            part_count = value
    input_path = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain, "sentences")
    output_path = os.path.join(os.getenv("OPIE_DIR"),
                               "data/domains", domain, "parse")
    cmd_string = "find " + output_path + " -size 0 | xargs rm -f"
    os.system(cmd_string)
    mkdir(output_path)
    parse_sentence(input_path, output_path, domain, part_count)
