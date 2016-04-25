#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/06 13:13:00

@author: Changzhi Sun
"""
import os
import sys
import getopt
import configparser

from my_package.static import Static
from my_package.static import get_wpath
from my_package.mysql_db import MysqlDB
from my_package.scripts import mkdir
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file


class PatternExtractor:

    def __init__(self, domain, mysql_db=None):
        self.table_opinwd = set(Static.opinwd.keys())
        self.table_profeat = set()
        self.mysql_db = mysql_db
        self.domain = domain
        self.domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                                       "data/domains", domain)
        self.pickle_parse = os.path.join(self.domain_dir,
                                         "pickles/parse_sentences")
        self.pickle_btsp = os.path.join(self.domain_dir,
                                        "pickles/bootstrapping")
        self.btsp_dir = os.path.join(self.domain_dir, "bootstrapping")
        self.p1_dep_type = set(["amod", "dep"])
        self.p2_dep_type = set(["acomp", "xcomp"])
        self.p3_dep_type = set(["advmod"])
        self.p4_mid_pos_tag = set(["PRP", "EX", "CD"])
        self.p4_coplua = set(["is", "was", "are", "were", "am", "be"])
        mkdir(self.btsp_dir)
        mkdir(self.pickle_btsp)

    def C1(self, sentence, idx_profeat, idx_opinwd, rname, profeat_pos_tag):
        if idx_opinwd not in sentence.dependency_tree:
            return
        for child in sentence.dependency_tree[idx_opinwd]:
            if child['type'] != "conj":
                continue
            idx_conj_opinwd = child['id']
            if not self.match_middle_and(sentence,
                                         idx_opinwd, idx_conj_opinwd):
                continue
            profeat, opinword, mark = self.match_all_condition(
                sentence, idx_profeat, idx_conj_opinwd,
                profeat_pos_tag, Static.ADJ | Static.ADV)
            if not mark:
                continue
            sentence.add_relation(profeat, opinword, rname)
            self.C2(sentence, profeat, idx_conj_opinwd, rname+"C2")

    def C2(self, sentence, profeat, idx_opinwd, rname):
        if idx_opinwd not in sentence.dependency_tree:
            return False
        if sentence.pos_tag[idx_opinwd] not in Static.ADJ:
            return False
        for child in sentence.dependency_tree[idx_opinwd]:
            if child['type'] != "xcomp":
                continue
            idx_verb = child['id']
            if sentence.pos_tag[idx_verb] not in Static.VERB:
                continue
            if sentence.is_weak_opinwd(sentence.tokens[idx_verb].lower()):
                continue
            if idx_verb not in sentence.dependency_tree:
                continue
            idx_to = None
            for ch in sentence.dependency_tree[idx_verb]:
                if ch['type'] == "aux":
                    idx_to = ch['id']
                    break
            if idx_to is None:
                continue
            if idx_to != idx_opinwd + 1 or idx_to != idx_verb - 1:
                continue
            opinwd = [idx_opinwd, idx_to, idx_verb]
            if not self.match_joint_condition(sentence, profeat, opinwd):
                continue
            sentence.add_relation(profeat, opinwd, rname)
            return True
        return False

    def C3(self, sentence, profeat, idx_profeat, idx_opinwd, rname):
        if profeat[0] - idx_opinwd == 1:
            opinwd = [idx_opinwd]
            opinwd.extend(profeat)
        elif idx_opinwd - profeat[-1] == 1:
            opinwd = profeat.copy()
            opinwd.append(idx_opinwd)
        else:
            return
        if len(opinwd) > 6:
            return
        for child in sentence.dependency_tree[idx_profeat]:
            if child['type'] != "nsubj":
                continue
            idx_nn_profeat = child['id']
            if not self.match_profeat_condition(sentence,
                                                idx_nn_profeat, Static.NOUN):
                continue
            nn_profeat = sentence.get_max_NP(idx_nn_profeat, idx_profeat)
            if not self.match_joint_condition(sentence, nn_profeat, opinwd):
                continue
            sentence.add_relation(nn_profeat, opinwd, rname)

    def P(self, sentence, dep_type, rname, profeat_pos_tag, opinwd_pos_tag):
        for idx_token, childs in sentence.dependency_tree.items():
            if idx_token == 0:
                continue
            for child in childs:
                if child['type'] not in dep_type:
                    continue
                idx_profeat, idx_opinwd = idx_token, child['id']
                profeat, opinword, mark = self.match_all_condition(
                    sentence, idx_profeat, idx_opinwd,
                    profeat_pos_tag, opinwd_pos_tag)
                if not mark:
                    continue
                sentence.add_relation(profeat, opinword, rname)
                self.C1(sentence, idx_profeat, idx_opinwd,
                        rname+"C1", profeat_pos_tag)
                self.C2(sentence, profeat, idx_opinwd, rname+"C2")
                if (profeat_pos_tag == Static.VERB and
                        opinwd_pos_tag == Static.ADV):
                    self.C3(sentence, profeat,
                            idx_profeat, idx_opinwd, rname+"C3")

    def P1(self, sentence):
        self.P(sentence, self.p1_dep_type, "P1", Static.NOUN, Static.ADJ)

    def P2(self, sentence):
        self.P(sentence, self.p2_dep_type, "P2", Static.VERB, Static.ADJ)

    def P3(self, sentence):
        self.P(sentence, self.p3_dep_type, "P3", Static.VERB, Static.ADV)

    def P4(self, sentence):
        for idx_token, childs in sentence.dependency_tree.items():
            if idx_token == 0:
                continue
            for child in childs:
                idx_profeat, idx_opinwd = idx_token, child['id']
                if idx_profeat > idx_opinwd:
                    continue
                if not self.match_middle_condition(sentence,
                                                   idx_profeat, idx_opinwd):
                    continue
                if not self.match_profeat_condition(sentence,
                                                    idx_profeat, Static.NOUN):
                    break
                profeat = sentence.get_max_NP(idx_profeat, idx_opinwd)
                if (sentence.pos_tag[idx_opinwd] in Static.NOUN and
                        self.match_opinwd_condition(sentence,
                                                    idx_opinwd, Static.NOUN)):
                    opinwd = sentence.get_max_NP(idx_opinwd, idx_profeat)
                elif (sentence.pos_tag[idx_opinwd] in Static.VERB and
                        self.match_opinwd_condition(sentence,
                                                    idx_opinwd, Static.VERB)):
                    opinwd = sentence.get_max_VP(idx_opinwd, idx_profeat)
                elif (sentence.pos_tag[idx_opinwd] in Static.ADJ and
                        self.match_opinwd_condition(sentence,
                                                    idx_opinwd, Static.ADJ)):
                    opinwd = sentence.get_max_ADJP(idx_opinwd, idx_profeat)
                else:
                    continue
                if not self.match_joint_condition(sentence, profeat, opinwd):
                    continue
                sentence.add_relation(profeat, opinwd, "P4")
                self.C1(sentence, idx_profeat,
                        idx_opinwd, "P4C1", Static.NOUN)

    def match_middle_and(self, sentence, idx1, idx2):
        idx_min, idx_max = min(idx1, idx2), max(idx1, idx2)
        for i in range(idx_min+1, idx_max):
            if sentence.tokens[i].lower() in set(["and", ","]):
                return True
        return False

    def match_middle_condition(self, sentence, idx_profeat, idx_opinwd):
        for i in range(idx_profeat+1, idx_opinwd):
            if sentence.pos_tag[i] in self.p4_mid_pos_tag:
                return False
        for i in range(idx_profeat+1, idx_opinwd):
            if sentence.tokens[i].lower() in self.p4_coplua:
                return True
        return False

    def match_profeat_condition(self, sentence, idx_profeat, pos_tag):
        profeat_str = sentence.tokens[idx_profeat].lower()
        if sentence.pos_tag[idx_profeat] not in pos_tag:
            return False
        if sentence.is_weak_profeat(profeat_str):
            return False
        if self.mysql_db and not self.mysql_db.language_model(profeat_str):
            return False
        return True

    def match_opinwd_condition(self, sentence, idx_opinwd, pos_tag):
        opinwd_str = sentence.tokens[idx_opinwd].lower()
        if opinwd_str not in self.table_opinwd:
            return False
        if sentence.pos_tag[idx_opinwd] not in pos_tag:
            return False
        if sentence.is_weak_opinwd(opinwd_str):
            return False
        if self.mysql_db and not self.mysql_db.language_model(opinwd_str):
            return False
        if (idx_opinwd > 1 and
                " ".join([sentence.tokens[idx_opinwd-1].lower(),
                          sentence.tokens[idx_opinwd]]) == "as well"):
            return False
        return True

    def match_joint_condition(self, sentence, profeat, opinwd):
        profeat_set, opinwd_set = set(profeat), set(opinwd)
        if profeat_set & opinwd_set:
            return False
        profeat, opinwd = tuple(profeat), tuple(opinwd)
        if (profeat, opinwd) in sentence.relation:
            return False
        return True

    def match_all_condition(self,
                            sentence,
                            idx_profeat,
                            idx_opinwd,
                            profeat_pos_tag,
                            opinwd_pos_tag):
            if not self.match_profeat_condition(sentence,
                                                idx_profeat, profeat_pos_tag):
                return None, None, False
            if not self.match_opinwd_condition(sentence,
                                               idx_opinwd, opinwd_pos_tag):
                return None, None, False
            if profeat_pos_tag == Static.NOUN:
                profeat = sentence.get_max_NP(idx_profeat, idx_opinwd)
            else:
                profeat = sentence.get_max_VP(idx_profeat, idx_opinwd)
            if not self.match_joint_condition(sentence, profeat, [idx_opinwd]):
                return None, None, False
            return profeat, [idx_opinwd], True

    def handle_sentence(self, sentence):
        self.P1(sentence)
        self.P2(sentence)
        self.P3(sentence)
        self.P4(sentence)

    def handle_sentences(self, sentences):
        for sentence in sentences:
            sentence.init_bootstrapping()
            self.handle_sentence(sentence)


def write_sentence(sentence, f):
    print("S\t{0}".format(sentence.text), file=f)
    for (profeat, opinwd), rname in sentence.relation.items():
        print("R\t{0}\t{1}\t{2}\t{3}\t{4}".format(
              sentence.print_phrase(profeat).lower(),
              sentence.print_phrase(opinwd).lower(),
              list(profeat), list(opinwd), rname),
              file=f)


def write(sentences, f):
    for sentence in sentences:
        if sentence.relation:
            write_sentence(sentence, f)


def usage():
    '''print help information'''
    print("pattern_extractor.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-b, --begin: parse_sentences pickel 文件的开始编号(包含此文件)")
    print("-e, --end: parse_sentences pickel 文件的结束编号(不包含此文件)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:",
                                   ["help", "domain=", "begin=", "end="])
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
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)

    r = PatternExtractor(domain)
    f = open(os.path.join(r.btsp_dir, "bootstrapping.relation"),
             "w", encoding="utf8")
    i = b
    spath = os.path.join(r.pickle_parse, "parse_sentences_%d.pickle" % i)
    while i < e and os.path.exists(spath + ".bz2"):
        print("pickle index: %d  loading" % i)
        sentences = load_pickle_file(spath)
        print("pickle index: %d  loaded" % i)
        r.handle_sentences(sentences)
        write(sentences, f)
        save_pickle_file(
            os.path.join(r.pickle_btsp,
                         "bootstrapping_sentences_%d.pickle" % i),
            [e for e in sentences if e.relation])
        i += 1
        spath = os.path.join(r.pickle_parse, "parse_sentences_%d.pickle" % i)
    f.close()
