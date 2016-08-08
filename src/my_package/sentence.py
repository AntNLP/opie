#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/05 19:06:42

@author: Changzhi Sun
"""
import os
import re

from nltk import Tree

from my_package.scripts import load_pickle_file
from my_package.static import Static
from normalization import remove_subset, merge_continues


class Sentence:

    def __init__(self):
        pass

    def set_text_score_review(self, text, score=None, review_index=None):
        self.text = text
        self.score = score
        self.review_index = review_index

    def set_parse_info(self, tokens, pos_tag, parse_string, dependency_tree):
        self.tokens = tokens
        self.pos_tag = pos_tag
        self.dependency_tree = dependency_tree
        self.parse_tree = Tree.fromstring(parse_string)
        for i in range(len(self.parse_tree.leaves())):
            self.parse_tree.__setitem__(
                self.parse_tree.leaf_treeposition(i), i+1)

    def get_all_phrase(self, idx, phrase):
        '''from parse_tree get all the PP of minimum that contains idx

        Keyword Argument:

        phrase -- "NP" "VP" or "ADJP"

        '''
        phrase_list = []
        index = self.parse_tree.leaf_treeposition(idx-1)
        n = len(index)
        for k in range(n-1, -1, -1):
            if k == n - 1 or self.parse_tree[index[:k]].label() == phrase:
                tmp = self.parse_tree[index[:k]].leaves()
                if tmp not in phrase_list:
                    phrase_list.append(tmp)
                if len(phrase_list) == 2:
                    break
        return phrase_list

    def get_min_phrase(self, idx1, idx2, phrase):
        phrase_list = self.get_all_phrase(idx1, phrase)
        if len(phrase_list) > 1 and idx2 not in phrase_list[1]:
            return phrase_list[1]
        return phrase_list[0]

    def get_max_phrase(self, idx1, idx2, phrase, word_len=8):
        phrase_list = self.get_all_phrase(idx1, phrase)
        for tmp in phrase_list[::-1]:
            if len(tmp) <= word_len and idx2 not in tmp:
                return tmp

    def get_min_NP(self, idx1, idx2=0):
        return self.get_min_phrase(idx1, idx2, "NP")

    def get_min_VP(self, idx1, idx2=0):
        return self.get_min_phrase(idx1, idx2, "VP")

    def get_min_ADJP(self, idx1, idx2=0):
        return self.get_min_phrase(idx1, idx2, "ADJP")

    def get_max_NP(self, idx1, idx2=0):
        return self.get_max_phrase(idx1, idx2, "NP")

    def get_max_VP(self, idx1, idx2=0):
        return self.get_max_phrase(idx1, idx2, "VP")

    def get_max_ADJP(self, idx1, idx2=0):
        return self.get_max_phrase(idx1, idx2, "ADJP")

    def init_bootstrapping(self):
        self.relation = {}

    def print_phrase(self, phrase):
        phrase_list = [self.tokens[i] for i in phrase]
        return " ".join(phrase_list)

    def add_relation(self, idx_profeat, idx_opinwd, rname):
        self.relation[(tuple(idx_profeat), tuple(idx_opinwd))] = rname

    def generate_candidate_relation(self, table_opinwd, mysql_db=None, complex_opinwd=None, test=False):
        if "candidate_relation" in dir(self):
            return
        self.candidate_dependency = []
        idx_candidate_profeat = self.generate_candidate_profeat()
        idx_candidate_opinwd = self.generate_candidate_opinwd()
        self.candidate_relation = self.restrict_candidate_relation(
            idx_candidate_profeat, idx_candidate_opinwd,
            table_opinwd, mysql_db, test, complex_opinwd)


    def generate_candidate_profeat(self):
        idx_candidate_profeat_set = self.generate_candidate_NP()
        idx_candidate_profeat_set |= self.generate_candidate_VP()
        return list(idx_candidate_profeat_set)

    def generate_candidate_phrase(self, phrase, pos_tag):
        n = len(self.tokens)
        idx_candidate_set = set()
        for i in range(1, n+1):
            if self.pos_tag[i] not in pos_tag:
                continue
            idx_candidate_set.add(tuple([i]))
            idx_candidate_set.add(tuple(self.get_min_phrase(i, 0, phrase)))
        return idx_candidate_set

    def generate_candidate_NP(self):
        return self.generate_candidate_phrase("NP", Static.NOUN)

    def generate_candidate_VP(self):
        return self.generate_candidate_phrase("VP", Static.VERB)

    def generate_candidate_opinwd(self):
        idx_candidate_opinwd_set = self.generate_candidate_VP()
        idx_candidate_opinwd_set |= self.generate_candidate_ADJP()
        idx_candidate_opinwd_set |= self.generate_candidate_ADV()
        #  for i in range(1, len(self.tokens) + 1):
            #  if self.tokens[i].lower() in table_opinwd:
                #  idx_candidate_opinwd_set.add(tuple([i]))
        return list(idx_candidate_opinwd_set)

    def generate_candidate_ADJP(self):
        return self.generate_candidate_phrase("ADJP", Static.ADJ)

    def generate_candidate_ADV(self):
        n = len(self.tokens)
        idx_candidate_set = set()
        for i in range(1, n+1):
            if self.pos_tag[i] not in Static.ADV:
                continue
            idx_candidate_set.add(tuple([i]))
        return idx_candidate_set

    def restrict_candidate_relation(self,
                                    idx_candidate_profeat,
                                    idx_candidate_opinwd,
                                    table_opinwd,
                                    mysql_db,
                                    test,
                                    complex_opinwd=None):
        idx_candidate_profeat = self.restrict_candidate_profeat(
            idx_candidate_profeat, mysql_db)
        idx_candidate_opinwd = self.restrict_candidate_opinwd(
            idx_candidate_opinwd, mysql_db, table_opinwd, complex_opinwd, test)
        return self.restrict_candidate_joint(idx_candidate_profeat,
                                             idx_candidate_opinwd,
                                             test, complex_opinwd)

    def restrict_candidate_joint(self,
                                 idx_candidate_profeat,
                                 idx_candidate_opinwd,
                                 test,
                                 complex_opinwd):
        self.dep_count = 20
        idx_candi_set = set()
        idx_candidate_relation = []
        for idx_profeat in idx_candidate_profeat:
            for idx_opinwd in idx_candidate_opinwd:
                if set(idx_profeat) & set(idx_opinwd):
                    continue
                if self.restrict_middle_word(idx_profeat, idx_opinwd):
                    continue
                idx_candi_set.add((idx_profeat, idx_opinwd))
        idx_candi_set = remove_subset(idx_candi_set)
        #  idx_candi_set = merge_continues(idx_candi_set)
        for idx_profeat, idx_opinwd in idx_candi_set:
            f, dist, dep_str = self.dependency_path(idx_profeat,
                                                    idx_opinwd)
            if not f or dist > self.dep_count:
                continue

            opinwd_str = self.print_phrase(idx_opinwd).lower()
            if complex_opinwd and opinwd_str in complex_opinwd:
                idx_candidate_relation.append((idx_profeat, idx_opinwd))
                self.candidate_dependency.append([dist, dep_str])
                continue

            # 训练集候选必须有一个包含在已经找到的 relation 中
            if not test:
                idx_profeat_set = set()
                idx_opinwd_set = set()
                for e1, e2 in self.relation.keys():
                    idx_profeat_set.add(e1)
                    idx_opinwd_set.add(e2)
                if (idx_profeat not in idx_profeat_set and
                        idx_opinwd not in idx_opinwd_set):
                    continue
            idx_candidate_relation.append((idx_profeat, idx_opinwd))
            self.candidate_dependency.append([dist, dep_str])
        # add relation
        #  for e in self.relation.keys():
            #  m = False
            #  for i in range(len(idx_candidate_relation)):
                #  if e == idx_candidate_relation[i]:
                    #  m = True
            #  if not m:
                #  f, dist, dep_str = self.dependency_path(e[0], e[1])
                #  idx_candidate_relation.append(e)
                #  self.candidate_dependency.append([dist, dep_str])
        return idx_candidate_relation

    def dependency_path(self, idx_profeat, idx_opinwd):
        idx_central_profeat = idx_profeat[-1]
        idx_central_opinwd = idx_opinwd[-1]
        for idx in idx_profeat:
            if self.pos_tag[idx] in (Static.NOUN | Static.VERB):
                idx_central_profeat = idx
                break
        for idx in idx_opinwd:
            if self.pos_tag[idx] in Static.SENTIMENT:
                idx_central_opinwd = idx
                break
        dependency_item = []
        profeat_to_root = self.root_to_current(idx_central_profeat)
        opinwd_to_root = self.root_to_current(idx_central_opinwd)
        b = min(len(profeat_to_root), len(opinwd_to_root))
        if profeat_to_root == opinwd_to_root[:b]:
            dependency_item.append(self.tokens[idx_central_profeat])
            dependency_item.append(self.pos_tag[idx_central_profeat])
            for e in opinwd_to_root[b:]:
                dependency_item.append(">")
                dependency_item.append(self.tokens[e])
                dependency_item.append(self.pos_tag[e])
                #  dependency_item.append(self.dependency_parent_type[e])

        elif profeat_to_root[:b] == opinwd_to_root:
            dependency_item.append(self.tokens[idx_central_profeat])
            dependency_item.append(self.pos_tag[idx_central_profeat])
            #  dependency_item.append(
            #  self.dependency_parent_type[idx_central_profeat])
            for e in profeat_to_root[::-1][1:]:
                dependency_item.append("<")
                dependency_item.append(self.tokens[e])
                dependency_item.append(self.pos_tag[e])
                if e == idx_central_opinwd:
                    break
                #  dependency_item.append(self.dependency_parent_type[e])
        else:
            for i in range(min(len(profeat_to_root), len(opinwd_to_root))):
                if profeat_to_root[i] != opinwd_to_root[i]:
                    b = i - 1
                    bb = profeat_to_root[i-1]
                    break
            dependency_item.append(self.tokens[idx_central_profeat])
            dependency_item.append(self.pos_tag[idx_central_profeat])
            #  dependency_item.append(self.dependency_parent_type[feat])
            for e in profeat_to_root[b:-1][::-1]:
                dependency_item.append("<")
                if e == 0:
                    dependency_item.append("ROOT")
                    dependency_item.append("ROOT")
                else:
                    dependency_item.append(self.tokens[e])
                    dependency_item.append(self.pos_tag[e])
                if e == bb:
                    break
                #  dependency_item.append(self.dependency_parent_type[e])
            for e in opinwd_to_root[b+1:]:
                dependency_item.append(">")
                dependency_item.append(self.tokens[e])
                dependency_item.append(self.pos_tag[e])
                #  dependency_item.append(self.dependency_parent_type[e])
        #  if None in set(dependency_item):
            #  return False, 0, 0
        return True, len(dependency_item), dependency_item

    def root_to_current(self, idx):
        if "dependency_parent" not in dir(self):
            self.get_dependency_parent()
        root_to_idx = [idx]
        if idx not in self.dependency_parent:
            return root_to_idx
        while(idx in self.dependency_parent and
                self.dependency_parent[idx] != 0):
            idx = self.dependency_parent[idx]
            root_to_idx.append(idx)
        root_to_idx.append(0)
        return root_to_idx[::-1]

    def get_dependency_parent(self):
        self.dependency_parent = {}
        self.dependency_parent[0] = 0
        self.dependency_parent_type = {}
        for idx_token, childs in self.dependency_tree.items():
            for child in childs:
                self.dependency_parent[child['id']] = idx_token
                self.dependency_parent_type[child['id']] = child['type']

    def restrict_middle_word(self, idx_profeat, idx_opinwd):
        idx_middle = self.get_middle_word(idx_profeat, idx_opinwd)
        if len(idx_middle) >= 7:
            return True
        it_set = set(["PRP", "EX"])
        and_set = set(["and", "but", "or", ",", ";"])
        for i in idx_middle:
            if self.pos_tag[i] in it_set:
                return True
            if self.tokens[i].lower() in and_set:
                return True
        return False

    def get_middle_word(self, idxes1, idxes2):
        max_value1 = max(idxes1)
        max_value2 = max(idxes2)
        if max_value1 < max_value2:
            min_value = min(idxes2)
            return list(range(max_value1+1, min_value))
        else:
            min_value = min(idxes1)
            return list(range(max_value2+1, min_value))

    def restrict_candidate_profeat(self, idx_candidate_profeat, mysql_db):
        idx_candidate = []
        for idx_profeat in idx_candidate_profeat:
            profeat_str = self.print_phrase(idx_profeat).lower()
            if len(idx_profeat) <= 7 and not self.is_weak_profeat(profeat_str):
                #  m = True
                #  for i in idx_profeat:
                    #  if self.is_weak_profeat(self.tokens[i].lower()):
                        #  m = False
                        #  break
                #  if m:
                idx_candidate.append(idx_profeat)
        return idx_candidate

    def restrict_candidate_opinwd(self,
                                  idx_candidate_opinwd,
                                  mysql_db,
                                  table_opinwd,
                                  complex_opinwd,
                                  test):
        idx_candidate = []
        for idx_opinwd in idx_candidate_opinwd:
            opinwd_str = self.print_phrase(idx_opinwd).lower()
            if complex_opinwd and opinwd_str in complex_opinwd:
                idx_candidate.append(idx_opinwd)
                continue
            #  if test:
                #  if (not self.is_weak_profeat(opinwd_str) and
                        #  len(idx_opinwd) <= 6):
                    #  idx_candidate.append(idx_opinwd)
            #  else:
            mark = False
            for e in idx_opinwd:
                if self.tokens[e].lower() in table_opinwd:
                    mark = True
                    break
            if not mark:
                continue

            if (not self.is_weak_opinwd(opinwd_str) and
                    len(idx_opinwd) <= 5):
                idx_candidate.append(idx_opinwd)
        return idx_candidate

    def generate_candidate_featvect_item(self, lexcion, test):
        if "feature_vector" in dir(self):
            return
        self.feature_vector_item = []
        for idx, idx_candidate_relation in enumerate(self.candidate_relation):
            self.feature_vector_item.append(self.create_relation_featvect(
                lexcion, idx, idx_candidate_relation, test))

    def create_relation_featvect(self,
                                 lexcion,
                                 idx,
                                 idx_candidate_relation,
                                 test):
        featvect = []
        idx_candidate_profeat, idx_candidate_opinwd = idx_candidate_relation

        # 0. 特征词和情感词的相对顺序
        relative_position = self.profeat_opinwd_order(
            idx_candidate_profeat, idx_candidate_opinwd)
        featvect.append([relative_position])
        # 1. 特征词前一个词
        featvect.append(
            self.create_one_word(lexcion, idx_candidate_profeat[0]-1,
                                 relative_position, True, True, 0, test))
        # 2. 特征词前两个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[0]-2,
                                             relative_position,
                                             True, True, 0, test))
        # 3. 特征词前一个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[0]-1,
                                             relative_position,
                                             True, True, 1, test))
        # 4. 特征词前两个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[0]-2,
                                             relative_position,
                                             True, True, 1, test))
        # 5. 特征词后一个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+1,
                                             relative_position,
                                             True, False, 0, test))
        # 6. 特征词后两个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+2,
                                             relative_position,
                                             True, False, 0, test))
        # 7. 特征词后一个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+1,
                                             relative_position,
                                             True, False, 1, test))
        # 8. 特征词后两个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+2,
                                             relative_position,
                                             True, False, 1, test))
        # 9. 情感词前一个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-1,
                                             relative_position,
                                             False, True, 0, test))
        # 10. 情感词前两个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-2,
                                             relative_position,
                                             False, True, 0, test))
        # 11. 情感词前一个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-1,
                                             relative_position,
                                             False, True, 1, test))
        # 12. 情感词前两个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-2,
                                             relative_position,
                                             False, True, 1, test))
        # 13. 情感词后一个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+1,
                                             relative_position,
                                             False, False, 0, test))
        # 14. 情感词后两个词
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+2,
                                             relative_position,
                                             False, False, 0, test))
        # 15. 情感词后一个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+1,
                                             relative_position,
                                             False, False, 1, test))
        # 16. 情感词后两个词 POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+2,
                                             relative_position,
                                             False, False, 1, test))

        idx_middle_word = self.get_middle_word(idx_candidate_profeat,
                                               idx_candidate_opinwd)
        # 17. 特征词和情感词中间词
        featvect.append(self.create_words(lexcion, idx_middle_word, 0, test))
        # 18. 特征词和情感词中间词 POS tag
        featvect.append(self.create_words(lexcion, idx_middle_word, 1, test))
        # 19. 特征词和情感词依赖树路径
        featvect.append(self.create_dependency_path(lexcion, idx, test))
        # 20. 特征词的 POS tag
        featvect.append(self.create_words(lexcion,
                                          idx_candidate_profeat,
                                          1, test))
        # 21. 情感词的 POS tag
        featvect.append(self.create_words(lexcion,
                                          idx_candidate_opinwd,
                                          1, test))
        # 22. 特征词
        featvect.append(self.create_words(lexcion,
                                          idx_candidate_profeat,
                                          0, test))
        # 23. 情感词
        featvect.append(self.create_words(lexcion,
                                          idx_candidate_opinwd,
                                          0, test))
        # 24. 依赖路径上词的个数
        featvect.append(self.create_dependency_count(lexcion, idx))
        # 25. 两个词之间是否有 is
        featvect.append(self.contain_BE(idx_middle_word))
        # 26. 特征词前一个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[0]-1,
                                             relative_position,
                                             True, True, 2, test))
        # 27. 特征词前两个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[0]-2,
                                             relative_position,
                                             True, True, 2, test))
        # 28. 特征词后一个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+1,
                                             relative_position,
                                             True, False, 2, test))
        # 29. 特征词后两个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_profeat[-1]+2,
                                             relative_position,
                                             True, False, 2, test))
        # 30. 情感词前一个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-1,
                                             relative_position,
                                             True, True, 2, test))
        # 31. 情感词前两个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[0]-2,
                                             relative_position,
                                             True, True, 2, test))
        # 32. 情感词后一个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+1,
                                             relative_position,
                                             True, False, 2, test))
        # 33. 情感词后两个词以及POS tag
        featvect.append(self.create_one_word(lexcion,
                                             idx_candidate_opinwd[-1]+2,
                                             relative_position,
                                             True, False, 2, test))
        # 34. 特征词和情感词中间词以及POS tag
        featvect.append(self.create_words(lexcion, idx_middle_word, 2, test))
        # 35. 特征词和情感词中间是否有this it ... [PRP, EX]
        featvect.append(self.contain_PRP_EX(idx_middle_word))
        # 36. 特征词和情感词的POS tag
        featvect.append(self.create_profeat_opinwd_pos_tag(
            lexcion, idx_candidate_profeat, idx_candidate_opinwd,
            len(idx_middle_word), test))
        # 37. 特征词的POS tags
        featvect.append(self.create_pos_tags(lexcion,
                                             idx_candidate_profeat,
                                             test))
        # 38. 情感词的POS tags
        featvect.append(self.create_pos_tags(lexcion,
                                             idx_candidate_opinwd,
                                             test))
        # 39. 特征词和情感词中间的POS tags
        featvect.append(self.create_pos_tags(lexcion, idx_middle_word, test))
        return featvect

    def contain_BE(self, idxes):
        BE = set(["is", "was", "are", "were", "am", "been", "be", "being"])
        for idx in idxes:
            if self.tokens[idx].lower() in BE:
                return [1]
        return [2]

    def contain_PRP_EX(self, idxes):
        it_set = set(["PRP", "EX"])
        for i in idxes:
            if self.pos_tag[i].lower() in it_set:
                return [1]
        return [2]

    def create_pos_tags(self, lexcion, idxes, test):
        lex = lexcion["unigram"]["pos_tags"]
        target = " ".join([self.pos_tag[e] for e in idxes])
        f = []
        if target not in lex:
            if not test:
                f.append(len(lex)+1)
                lex[target] = f[0]
        else:
            f.append(lex[target])
        return f

    def create_profeat_opinwd_pos_tag(self,
                                      lexcion,
                                      idx_profeat,
                                      idx_opinwd,
                                      num,
                                      test):
        profeat_pos_tag = " ".join([self.pos_tag[e] for e in idx_profeat])
        opinwd_pos_tag = " ".join([self.pos_tag[e] for e in idx_opinwd])
        joint_pos_tag = "#".join([profeat_pos_tag, opinwd_pos_tag])
        joint_pos_tag += "#%d" % num
        f = []
        lex = lexcion["unigram"]["joint_pos_tag"]
        if joint_pos_tag not in lex:
            if not test:
                f.append(len(lex)+1)
                lex[joint_pos_tag] = f[0]
        else:
            f.append(lex[joint_pos_tag])
        return f

    def profeat_opinwd_order(self, idx_profeat, idx_opinwd):
        if idx_profeat[-1] < idx_opinwd[0]:
            return 1
        else:
            return 2

    def create_one_word(self,
                        lexcion,
                        idx_token,
                        relative_position,
                        is_profeat,
                        is_left,
                        option,
                        test=False):
        '''
        @option: 0   word
                 1   pos_tag
                 2   word_pos_tag
        '''
        f = []
        if relative_position == 1 and is_profeat and not is_left:
            return f
        if relative_position == 2 and is_profeat and is_left:
            return f
        if relative_position == 1 and not is_profeat and is_left:
            return f
        if relative_position == 2 and not is_profeat and not is_left:
            return f
        if option == 0:
            target = ("#" if idx_token not in self.tokens
                      else self.tokens[idx_token].lower())
            lex = lexcion["unigram"]["word"]
            if target in Static.stopwords:
                return f
        elif option == 1:
            target = ("#" if idx_token not in self.tokens
                      else self.pos_tag[idx_token].lower())
            lex = lexcion["unigram"]["pos_tag"]
        else:
            target = ("##" if idx_token not in self.tokens
                      else " ".join([self.tokens[idx_token].lower(),
                                     self.pos_tag[idx_token].lower()]))
            lex = lexcion["unigram"]["word_pos_tag"]
        if re.search(r"^\W+$", target) != None:
            return f
        if target not in lex:
            if test:
                return f
            f.append(len(lex) + 1)
            lex[target] = f[0]
        else:
            f.append(lex[target])
        return f

    def create_words(self, lexcion, idxes, option, test=False):
        '''
        @option: 0   word
                 1   pos_tag
                 2   word_pos_tag
        '''
        f = []
        for w in idxes:
            if option == 0:
                target = self.tokens[w].lower()
                lex = lexcion["unigram"]["word"]
                if target in Static.stopwords:
                    continue
            elif option == 1:
                target = self.pos_tag[w].lower()
                lex = lexcion["unigram"]["pos_tag"]
            else:
                target = " ".join([self.tokens[w].lower(),
                                   self.pos_tag[w].lower()])
                lex = lexcion["unigram"]["word_pos_tag"]
            if re.search(r"^\W+$", target) != None:
                continue
            if target not in lex:
                if test:
                    continue
                f.append(len(lex) + 1)
                lex[target] = f[-1]
            else:
                f.append(lex[target])
        return f

    def create_dependency_path(self, lexcion, idx_dep, test=False):
        f = []
        dep_list = self.candidate_dependency[idx_dep][1]
        #  dep_str = "".join(dep_list).lower()
        #  if lexcion["unigram"]["dep"].get(dep_str) == None:
            #  if test:
                #  return f
            #  f.append(len(lexcion["unigram"]["dep"]) + 1)
            #  lexcion["unigram"]["dep"][dep_str] = f[0]
        #  else:
            #  f.append(lexcion["unigram"]["dep"][dep_str])
        #  return f
        for e in dep_list:
            e_low = e.lower()
            if e_low not in lexcion["unigram"]["dep"]:
                if test:
                    continue
                f.append(len(lexcion["unigram"]["dep"]) + 1)
                lexcion["unigram"]["dep"][e_low] = f[-1]
            else:
                f.append(lexcion["unigram"]["dep"][e_low])
        return f

    def create_dependency_count(self, lexcion, idx_dep):
        return [self.candidate_dependency[idx_dep][0]]


    def have_overlap(self, e1, e2):
        if set(e1[0]) & set(e2[0]) != set() and set(e1[1]) & set(e2[1]) != set():
            return True
        return False

    def generate_label(self, test):
        if test:
            self.label = [0] * len(self.candidate_relation)
            return
        self.label = []
        for each in self.candidate_relation:
            m = 0
            for key in self.relation.keys():
                if self.have_overlap(each, key):
                    m = 1
                    break
            self.label.append(m)
            #  if each in self.relation:
                #  self.label.append(1)
            #  else:
                #  self.label.append(0)

    def generate_candidate_featvect(self, lexcion):
        feature_vector = []
        word_len = len(lexcion["unigram"]["word"])
        pos_tag_len = len(lexcion["unigram"]["pos_tag"])
        word_pos_tag_len = len(lexcion["unigram"]["word_pos_tag"])
        pos_tags_len = len(lexcion["unigram"]["pos_tags"])
        idx_word = [

            #  特征词前一个词
            1,
            #  特征词前两个词
            2,
            #  特征词后一个词
            5,
            #  特征词后两个词
            6,
            #  情感词前一个词
            9,
            #  情感词前两个词
            10,
            #  情感词后一个词
            13,
            #  情感词后两个词
            14,
            # 特征词和情感词中间词
            17
            # 特征词
            #  22
            # 情感词
            #  23
            ]
        idx_pos_tag = [
            #  特征词前一个词 POS tag
            3,
            #  特征词前两个词 POS tag
            4,
            #  特征词后一个词 POS tag
            7,
            #  特征词后两个词 POS tag
            8,
            #  情感词前一个词 POS tag
            11,
            #  情感词前两个词 POS tag
            12,
            #  情感词后一个词 POS tag
            15,
            #  情感词后两个词 POS tag
            16,
            #  特征词和情感词中间词 POS tag
            18,
            #  特征词 POS tag
            20,
            #  情感词 POS tag
            21
            ]
        idx_word_pos_tag = [
            # 特征词前一个词以及POS tag
            26,
            # 特征词前两个词以及POS tag
            27,
            # 特征词后一个词以及POS tag
            28,
            # 特征词后两个词以及POS tag
            29,
            # 情感词前一个词以及POS tag
            30,
            # 情感词前两个词以及POS tag
            31,
            # 情感词后一个词以及POS tag
            32,
            # 情感词后两个词以及POS tag
            33,
            # 特征词和情感词中间词以及POS tag
            34
            ]
        idx_pos_tags = [
            # 特征词的POS tags
            37,
            # 情感词的POS tags
            38,
            # 特征词和情感词中间的POS tags
            39
            ]
        for f in self.feature_vector_item:
            base = 0
            feat_vector = []
            for i in idx_word:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += word_len

            for i in idx_pos_tag:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += pos_tag_len

            for i in idx_word_pos_tag:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += word_pos_tag_len

            for i in idx_pos_tags:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += pos_tags_len

            #  特征词和情感词依赖树路径
            for e in sorted(set(f[19])):
                feat_vector.append(base + e)
            base += len(lexcion["unigram"]["dep"])

            # 依赖路径上词的个数
            for e in f[24]:
                feat_vector.append(base + e)
            base += self.dep_count

            # 两词之间是否有be
            for e in f[25]:
                feat_vector.append(base + e)
            base += 2

            # 特征词和情感词中间是否有this it ... [PRP, EX]
            for e in f[35]:
                feat_vector.append(base + e)
            base += 2

            # 特征词和情感词的POS tag
            for e in sorted(set(f[36])):
                feat_vector.append(base + e)
            base += len(lexcion["unigram"]["joint_pos_tag"])

            # 特征词和情感词的相对顺序
            for e in f[0]:
                feat_vector.append(base + e)
            base += 2

            feature_vector.append(feat_vector)
        return feature_vector

    # relation data helper
    def generate_each_sentence(self):
        n = len(self.tokens)
        X, y = [], []
        for (profeat_idx, opinwd_idx), label, (dep_len, dep_str) in zip(self.candidate_relation,
                                                                        self.label,
                                                                        self.candidate_dependency):
            features = []
            if label == 1:
                y.append([0.0, 1.0])
            else:
                y.append([1.0, 0.0])
            if profeat_idx[-1] < opinwd_idx[0]:
                min_idx = profeat_idx
                max_idx = opinwd_idx
                features.append([1])
            else:
                min_idx = opinwd_idx
                max_idx = profeat_idx
                features.append([0])

            # L
            t = ["<S/>"]
            t.extend([self.tokens[j].lower() for j in range(1, min_idx[0])])
            features.append(t)
            t = ["<S/>"]
            t.extend([self.pos_tag[j] for j in range(1, min_idx[0])])
            features.append(t)

            # Min
            t = [self.tokens[j].lower() for j in min_idx]
            features.append(t)
            t = [self.pos_tag[j] for j in min_idx]
            features.append(t)

            # B
            t = [self.tokens[j].lower()
                 for j in range(min_idx[-1]+1, max_idx[0])]
            features.append(t)
            t = [self.pos_tag[j] for j in range(min_idx[-1]+1, max_idx[0])]
            features.append(t)

            # Max
            t = [self.tokens[j].lower() for j in max_idx]
            features.append(t)
            t = [self.pos_tag[j] for j in max_idx]
            features.append(t)

            # R
            t = [self.tokens[j].lower() for j in range(max_idx[-1]+1, n+1)]
            t.append("<E/>")
            features.append(t)
            t = [self.pos_tag[j] for j in range(max_idx[-1]+1, n+1)]
            t.append("<E/>")
            features.append(t)

            i = 0
            t1, t2, t3 = [], [], []
            while i < len(dep_str):
                t1.append(dep_str[i].lower())
                t2.append(dep_str[i+1])
                if i < len(dep_str) - 2:
                    t3.append(dep_str[i+2])
                i += 3
            t3.append(">")
            features.append(t1)
            features.append(t2)
            features.append(t3)

            #  t = [self.tokens[j].lower() for j in range(1, len(self.tokens)+1)]
            #  features.append(t)

            #  t = [self.pos_tag[j] for j in range(1, len(self.tokens)+1)]
            #  features.append(t)

            X.append(features)
        return X, y

    def generate_each_sentence_with_conv(self, w=5):
        n = len(self.tokens)
        X, y = [], []
        for (profeat_idx, opinwd_idx), label, (dep_len, dep_str) in zip(self.candidate_relation,
                                                                        self.label,
                                                                        self.candidate_dependency):
            features = []
            if label == 1:
                y.append([0.0, 1.0])
            else:
                y.append([1.0, 0.0])
            if profeat_idx[-1] < opinwd_idx[0]:
                min_idx = profeat_idx
                max_idx = opinwd_idx
                features.append([1])
            else:
                min_idx = opinwd_idx
                max_idx = profeat_idx
                features.append([0])

            t = [self.tokens[j].lower() for j in range(max(1, min_idx[0]-w), min_idx[0])]
            features.append(t)
            t = [self.pos_tag[j] for j in range(max(1, min_idx[0]-w), min_idx[0])]
            features.append(t)

            t = [self.tokens[j].lower() for j in range(min_idx[-1]+1, max_idx[0])]
            features.append(t)
            t = [self.pos_tag[j] for j in range(min_idx[-1]+1, max_idx[0])]
            features.append(t)

            t = [self.tokens[j].lower() for j in range(max_idx[-1]+1, min(max_idx[-1]+w+1, n+1))]
            features.append(t)
            t = [self.pos_tag[j] for j in range(max_idx[-1]+1, min(max_idx[-1]+w+1, n+1))]
            features.append(t)

            t = [self.tokens[j].lower() for j in min_idx]
            features.append(t)
            t = [self.pos_tag[j] for j in min_idx]
            features.append(t)

            t = [self.tokens[j].lower() for j in max_idx]
            features.append(t)
            t = [self.pos_tag[j] for j in max_idx]
            features.append(t)

            i = 0
            t1, t2 = [], []
            while i < len(dep_str):
                t1.append(dep_str[i].lower())
                t2.append(dep_str[i+1])
                #  if i < len(dep_str) - 2:
                    #  t.append((dep_str[i+2], dep_str[i+2]))
                i += 3
            features.append(t1)
            features.append(t2)
            X.append(features)
        return X, y

    @classmethod
    def is_weak(cls, word_str, table_weak):
        if word_str in table_weak:
            return True
        if re.search(r"^\W*$", word_str) != None:
            return True
        return False

    @classmethod
    def is_weak_profeat(cls, profeat_str):
        return cls.is_weak(profeat_str, Static.weak_profeat)

    @classmethod
    def is_weak_opinwd(cls, opinwd_str):
        return cls.is_weak(opinwd_str, Static.weak_opinwd)

if __name__ == "__main__":
    #  parse_sentence_path = os.path.join(
        #  os.getenv("OPIE_DIR"),
        #  "data/domains/reviews_Musical_Instruments/
        #  pickles/parse_sentences/parse_sentences_1.pickle")
    #  sentences = load_pickle_file(parse_sentence_path)
    #  for sentence in sentences:
        #  print(sentence.text)
        #  print(sentence.dependency_tree)
        #  print(sentence.dependency_path([1, 2], [10, 11]))
        #  break
    #  print("end")
    pass
