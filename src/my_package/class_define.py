# -*- coding: utf-8 -*-
'''
Created on 2015年8月28日

@author: Changzhi Sun
'''
import nltk
from nltk import Tree
from nltk.corpus import stopwords
import re
from nltk.tokenize import TreebankWordTokenizer
from my_package.scripts import load_file_line, return_none, all_match, all_cover, have_part
from _collections import defaultdict
from itertools import chain

class Sentence:
    def __init__(self):
        '''

        initialize

        '''
        self.feats = None #规则产生的特征词(类) [[begin,..., end], ...]
        self.feats_regu = None #产生特征词所用的规则
        self.feats_set = None #规则产生的特征词集合{(begin, ... end), ...}

        self.sents = None #规则产生的情感词 [[(b, ..., e), polarity], ...]
        self.sents_regu = None # 产生情感词所用的规则
        self.sents_dict = None #规则产生的情感词集合{(b, ... e), ...}

        self.feature_sentiment = None #[[(), (), polarity], ....]
        self.fs_regu = None #产生 pair 对所用的规则
        self.fs_dict = None #产生 feature-sentiment 的结合 {(特征词索引， 情感词索引), ...}


        self.text = None #句子的内容

        self.tokens = None #句子的分词结果，下标从 1 开始

        self.pos_tag = None #句子的POS tag， 下标从 1 开始

        self.parse_string = None #句子的解析树 string

        self.dependency_tree = None #句子的依赖树

        self.score = None #该句子所在 review 的评分

        self.polarity = None #该句子的极性

        self.review_index =  None #该句子所在 review 的index

        self.parse_tree = None #该句子的 parse tree

        self.dictionary_of_np = None #

        self.dictionary_of_vp = None #

        self.dictionary_of_adjp = None #

        self.dependency_tree_up = None #

        self.all_sentiment = None #

        self.all_vp = None #

        self.all_np = None #

        self.sentiment = None #

        self.text_and_pos_tag = None #

        self.feature_vector = None

        self.candidate_pairs = None

    def inquire_content(self, connection, var, table_lm, t=-25):
        try:

            # 游标
            with connection.cursor() as cursor:
                sql = "select * from {0} where content=\"{1}\" and score>={2}".format(table_lm, var, t)
                #  sql = "select * from lm_db where content=%s"
                #  sql = "select * from lm_db where content=\"{0}\"".format(var)
                #  cursor.execute(sql, (var))
                cursor.execute(sql)
                res = cursor.fetchall()
                if len(res) == 0:
                    return False
                else:
                    return True
        except Exception as err:
            print(err)
            print(var)
            return False
        finally:
            pass

    def fs_add(self, feature_sentiment_ele, regu):
        '''添加元素到 feature-sentiment 以及 fs_regu 中
        '''
        #feature_sentiment_ele -- (feature, sentiment, flag)
        #regu -- (name, single, down1, type1, up, type2, down2)
        # 当所用规则是 "f_using_s2" 时，需要执行合并操作
        #  if regu[0] == "f_using_s2":
            #  self.merge(feature_sentiment_ele)
        self.feature_sentiment.append(feature_sentiment_ele)
        self.fs_regu.append(regu)



    def merge(self, feature_sentiment_ele):
        ''''''
        n = len(self.feature_sentiment)
        i = 0
        while i < n:
            if (self.feature_sentiment[i][0][0] in feature_sentiment_ele[1]
                and self.feature_sentiment[i][1][0] in feature_sentiment_ele[1]) or (
                self.feature_sentiment[i][0] == feature_sentiment_ele[0] and
                self.feature_sentiment[i][1][0] in feature_sentiment_ele[1]):
                del(self.feature_sentiment[i])
                del(self.fs_regu[i])
                n -= 1
                i -= 1
            i += 1

    def set_init(self):
        '''对相应类型的数据初始化
        '''
        if self.sents == None:
            self.sents = []
        if self.sents_dict == None:
            self.sents_dict = defaultdict(return_none)
        if self.sents_regu == None:
            self.sents_regu = []

        if self.feature_sentiment == None:
            self.feature_sentiment = []
        if self.fs_regu == None:
            self.fs_regu = []
        if self.fs_dict == None:
            self.fs_dict = {}

        if self.feats_set == None:
            self.feats_set = set()
        if self.feats == None:
            self.feats = []
        if self.feats_regu == None:
            self.feats_regu = []

        self.dictionary_of_adjp = self.get("ADJP")

    def set_text_score_review(self, text, score, review_index):
        '''置句子的 text 以及 score '''
        self.text, self.score, self.review_index = text, score, review_index

        self.polarity = 1 if self.score >= 3 else -1

    def set_parse_info(self, tokens, pos_tag, parse_string, dependency_tree):
        '''置句子的  parse 信息 '''
        self.tokens, self.pos_tag, self.parse_string, self.dependency_tree = tokens, pos_tag, parse_string, dependency_tree

        self.parse_tree = Tree.fromstring(self.parse_string)
        for i in range(len(self.parse_tree.leaves())):
            self.parse_tree.__setitem__(self.parse_tree.leaf_treeposition(i), i + 1)

        self.dictionary_of_np = self.get("NP")
        self.dictionary_of_vp = self.get("VP")
        self.dictionary_of_adjp = self.get("ADJP")


        self.get_dependence_tree_up()
        self.text_and_pos_tag = ""
        for key in self.tokens:
            self.text_and_pos_tag += self.tokens[key] + " [" + self.pos_tag[key] + "] "

    def get(self, PP):
        '''from parse_tree get all the PP of minimum that contains k

        Keyword Argument:

        PP -- "NP" or "VP"

        '''
        ret_dict = {}
        for i in range(len(self.parse_tree.leaves())):
            ret_dict[i+1] = []
            index = self.parse_tree.leaf_treeposition(i)
            n = len(index)
            for k in range(n-1, -1, -1):
                if k == n - 1 or self.parse_tree[index[:k]].label() == PP:
                    temp = self.parse_tree[index[:k]].leaves()
                    if temp not in ret_dict[i+1]:
                        ret_dict[i+1].append(temp)
        return ret_dict


    def get_np(self, k, m = -1):
        '''get NP of minimum that contains the token-offset of the k

        Keyword argument:

        k -- the token-offset of the k
        m -- the other token-offset of the k

        '''
        ret_list = [k]
        if len(self.dictionary_of_np[k]) > 1:
            ret_list = self.dictionary_of_np[k][1]
        if m != -1 and m in ret_list:
            ret_list = [k]
        return ret_list

    def get_vp(self, k, m = -1):
        '''get VP of minimum that contains the token-offset of the k

        Keyword argument:

        k -- the token-offset of the k
        m -- the other token-offset of the k

        '''
        ret_list = [k]
        if len(self.dictionary_of_vp[k]) > 1:
            ret_list = self.dictionary_of_vp[k][1]
        #  the VP can't contains the BE
        for l in ret_list:
            if self.tokens[l].lower() in Static.BE:
                ret_list = [k]
                break
        #  the VP can't contains the sentiment
        if m != -1 and m in ret_list:
            ret_list = [k]
        if len(ret_list) > 10:
            ret_list = [k]
        return ret_list

    def get_min_adjp(self, k, a=[]):
        self.dictionary_of_adjp = self.get("ADJP")
        if a != []:
            a = set(a)
            if len(self.dictionary_of_adjp[k]) > 1 and a & set(self.dictionary_of_adjp[k][1]):
                return self.dictionary_of_adjp[k][0]
            else:
                return self.dictionary_of_adjp[k][1]
        else:
            if len(self.dictionary_of_adjp[k]) > 1:
                return self.dictionary_of_adjp[k][1]
        return self.dictionary_of_adjp[k][0]

    def get_max_adjp(self, k, a=[]):
        self.dictionary_of_adjp = self.get("ADJP")
        if a != []:
            a = set(a)
            for e in self.dictionary_of_adjp[k][::-1]:
                if a & set(e) == set():
                    return e
        return self.dictionary_of_adjp[k][-1]

    def get_dependence_tree_up(self):
        ''''''
        self.dependency_tree_up = {}
        self.dependency_tree_up[0] = 0
        self.dependency_tree_up_type = defaultdict(return_none)
        for key, values in self.dependency_tree.items():
            if values != None:
                for value in values:
                    self.dependency_tree_up[value['id']] = key
                    self.dependency_tree_up_type[value['id']] = value['type']

    def judge_as_well(self, k):
        '''判断当前词是否是 as well '''
        if k == 1:
            return False
        if self.tokens[k].lower() == "well" and self.tokens[k-1].lower() == "as":
            return True
        return False

    def build_trie(self, trie):
        tokens = self.text.lower().split(' ')
        self.trie = Trie()
        for i in range(len(tokens)):
            #  trie.add(tokens[i:])
            self.trie.add(tokens[i:])


    def get_all_sentiment(self, sentiments):
        '''得到所有的情感词'''
        n = len(self.pos_tag)
        self.all_sentiment = []
        for i in range(1, n+1):
            #  情感词词性
            if self.pos_tag[i] not in Static.JJ and self.pos_tag[i] not in Static.RB:
                continue
            word_str = self.tokens[i].lower()

            #  情感词典
            if word_str not in sentiments:
                continue

            #  去除 "as well"
            if self.judge_as_well(i):
                continue

            self.all_sentiment.append([i])

        for i in range(1, n+1):
            if self.pos_tag[i] not in Static.VB:
                continue
            i_word = self.tokens[i].lower()
            for j in range(1, n+1):
                if self.pos_tag[j] not in Static.RB:
                    continue
                ij_word = i_word + " " + self.tokens[j].lower()
                if ij_word not in sentiments:
                    continue
                self.all_sentiment.append([i, j])

        np_set = set()
        for i in range(1, n+1):
            if self.pos_tag[i] not in Static.NN:
                continue
            np = self.get_np(i)
            if len(np) == 1:
                continue
            np_string = self.get_phrase(np).lower()

            f_x = False
            for x in np[:-1]:
                if self.pos_tag[x] in Static.NN:
                    f_x = True
                if self.pos_tag[x] in Static.JJ:
                    f_x = True
                if self.pos_tag[x] in Static.RB:
                    f_x = True
                if self.pos_tag[x] in Static.VB:
                    f_x = True

            if not f_x:
                continue
            if np_string not in sentiments:
                continue
            np_set.add(tuple(np))
        self.all_sentiment.extend([list(e) for e in np_set])

    def root_to_this(self, this):
        '''返回从根节点到this依赖路径的列表

        keyword argument:

        this -- 当前节点索引

        '''
        ret_list =[this]
        if self.dependency_tree_up.get(this, None) == None:
            return [this]
        while self.dependency_tree_up.get(this) != None and self.dependency_tree_up[this] != 0:
            this = self.dependency_tree_up[this]
            ret_list.append(this)
        ret_list.append(0)
        return ret_list[::-1]

    def path_word(self, begin, end):
        '''返回依赖树上begin和end路径上所有的词

        keyword argument:

        begin -- 特征词下标
        end -- 情感词下标

        '''
        feat, sent = None, None

        for e in begin:
            if self.pos_tag[e] in Static.NN or self.pos_tag[e] in Static.VB:
                feat = e
                break
        for e in end:
            if self.pos_tag[e] in Static.NN:
                sent = e
                break
        if sent == None:
            for e in end:
                if self.pos_tag[e] in Static.SENTIMENT:
                    sent = e
                    break

        if feat == None or sent == None:
            return False, 0, 0
        #  print(self.get_phrase(begin), self.get_phrase(end))
        #  print("feat:", feat, " sent:", sent)
        #  print(self.tokens[feat], self.tokens[sent])
        ret_dep_direct = []
        #  dep_order = []
        list1 = self.root_to_this(feat)
        list2 = self.root_to_this(sent)
        b = min(len(list1), len(list2))
        #  print(list1)
        #  print(list2)

        if list1 == list2[:b]:
            ret_dep_direct.append(self.tokens[feat])
            #  ret_dep_direct.append(self.pos_tag[feat])
            for e in list2[b:]:
                ret_dep_direct.append(">")
                #  dep_order.append(">")
                ret_dep_direct.append(self.tokens[e])
                #  ret_dep_direct.append(self.pos_tag[e])
                #  ret_dep_direct.append(self.dependency_tree_up_type[e])

        elif list1[:b] == list2:
            ret_dep_direct.append(self.tokens[feat])
            #  ret_dep_direct.append(self.pos_tag[feat])
            ret_dep_direct.append(self.dependency_tree_up_type[feat])
            for e in list1[::-1][1:]:
                ret_dep_direct.append("<")
                #  dep_order.append("<")
                ret_dep_direct.append(self.tokens[e])
                #  ret_dep_direct.append(self.pos_tag[e])
                if e == sent:
                    break
                #  ret_dep_direct.append(self.dependency_tree_up_type[e])
        else:
            for i in range(min(len(list1), len(list2))):
                if list1[i] != list2[i]:
                    b = i - 1
                    bb = list1[i-1]
                    break
            ret_dep_direct.append(self.tokens[feat])
            #  ret_dep_direct.append(self.pos_tag[feat])
            ret_dep_direct.append(self.dependency_tree_up_type[feat])
            for e in list1[b:-1][::-1]:
                ret_dep_direct.append("<")
                #  dep_order.append("<")
                ret_dep_direct.append(self.tokens[e])
                #  ret_dep_direct.append(self.pos_tag[e])
                if e == bb:
                    break
                #  ret_dep_direct.append(self.dependency_tree_up_type[e])


            for e in list2[b+1:]:
                ret_dep_direct.append(">")
                #  dep_order.append(">")

                ret_dep_direct.append(self.tokens[e])
                #  ret_dep_direct.append(self.pos_tag[e])
                #  ret_dep_direct.append(self.dependency_tree_up_type[e])

        #  print(ret_dep_direct)
        if None in set(ret_dep_direct):
            return False, 0, 0
        return True, len(ret_dep_direct), ret_dep_direct


    def get_phrase(self, index=[]):
        '''get phrase of the sentence by index, and index is number sequence

        Keyword argument:

        index -- the sequence of number

        '''
        if index == []:
            index = [i for i in range(1, len(self.pos_tag) + 1)]
        phrase = [self.tokens[word] for word in index]
        return " ".join(phrase)

    def check_not_one(self, i):
        '''count the number of contrary word in [i-5, i+5]

        Keyword Argument:

        i -- the token-offset of the sentence

        '''
        count, k = 0 , i - 1 #the number of contrary word

        #the token-offset of sentence k must no smaller than 0
        while k > 0 and k >= i - 5:
            if self.tokens[k].lower() in Static.terminal_signal:
                break
            if self.tokens[k].lower() in Static.contrary:
                count += 1
            k -= 1
        k, l = i + 1, len(self.tokens)

        #the token-offset of sentence k must no larger than the length of sentence
        while k <= l and k <= i + 5:
            if self.tokens[k].lower() in Static.terminal_signal:
                break
            if self.tokens[k].lower() in Static.contrary:
                count += 1
            k += 1
        return count


    def check_not(self, a, b = -1):
        ''' check not

        Keyword argument:

        a -- the index
        b -- the other index

        '''
        if b == -1:
            return self.check_not_one(a)
        count = 0
        max_value, min_value = a, b
        if b > a:
            max_value, min_value = b, a
        for k in range(min_value, max_value + 1):
            if self.tokens[k].lower() in Static.contrary:
                count += 1
        return count

    def is_weak_feature(self, word):
        '''判断当前特征词 是否 weak
        '''
        for line in Static.weak_feature:
            if line in set(word.split()):
                return True
        if re.search(r"^\W*$", word) != None:
            return True
        return False

    def get_all_np(self, connection, table_lm):
        '''get all NP
        '''
        n = len(self.pos_tag)
        self.all_np = set()
        for i in range(1, n+1):
            if self.pos_tag[i] not in Static.NN:
                continue
            s_word = self.tokens[i].lower()
            if len(self.dictionary_of_np[i]) > 1:
                pp = tuple(self.dictionary_of_np[i][1])
                pp_string = self.get_phrase(pp).lower()
                if self.inquire_content(connection, pp_string, table_lm) and not not self.is_weak_feature(pp_string):
                    self.all_np.add(pp)
            if self.inquire_content(connection, self.tokens[i].lower(), table_lm) and not self.is_weak_feature(s_word):
                self.all_np.add(tuple([i]))
        self.all_np = list(self.all_np)

    def get_all_vp(self, connection, table_lm):
        '''get all VP
        '''
        n = len(self.pos_tag)
        self.all_vp = set()
        for i in range(1, n+1):
            if self.pos_tag[i] in Static.VB and self.tokens[i].lower() not in Static.BE:
                s_word = self.tokens[i].lower()
                if len(self.dictionary_of_vp[i]) > 1:
                    pp = tuple(self.dictionary_of_vp[i][1])
                    pp_string = self.get_phrase(pp).lower()
                    if self.inquire_content(connection, pp_string, table_lm) and not not self.is_weak_feature(pp_string):
                        self.all_vp.add(pp)
                if self.inquire_content(connection, self.tokens[i].lower(), table_lm) and not self.is_weak_feature(s_word):
                        self.all_vp.add(tuple([i]))
        self.all_vp = list(self.all_vp)

    def get_between_word(self, key1, key2):
        '''
        keyword argument:

        key1 -- 当前的一个索引
        key2 -- 当前的另一个索引

        '''
        maxValue1 = max(key1)
        maxValue2 = max(key2)
        if maxValue1 < maxValue2:
            minValue = min(key2)
            return list(range(maxValue1+1, minValue))
        else:
            minValue = min(key1)
            return list(range(maxValue2+1, minValue))

    def generate_candidate(self, sentiments, connection, table_lm, test=False):
        '''生成所有的候选特征-情感词对'''
        if self.candidate_pairs != None:
            return
        self.get_dependence_tree_up()
        self.get_all_np(connection, table_lm)
        self.get_all_vp(connection, table_lm)
        self.get_all_sentiment(sentiments)

        if test == False:
            new_set = set()
            for feat, sent in self.feature_sentiment:
                if len(sent) == 3:
                    new_set.add(tuple(sent))
            self.all_sentiment.extend([list(e) for e in new_set])
            fset = set([tuple(e[0]) for e in self.feature_sentiment])
            sset = set([tuple(e[0]) for e in self.feature_sentiment])
        pp = self.all_np.copy()
        pp.extend(self.all_vp)
        self.candidate_pairs_dependency_dist = []
        self.candidate_pairs = []
        for e1 in pp:
            for e2 in self.all_sentiment:
                if not test and tuple(e1) not in fset and tuple(e2) not in sset:
                    continue
                #  if len(self.get_between_word(e1, e2)) >= 10:
                    #  continue
                if set(e1) & set(e2) != set():
                    continue
                f, dist, dep_str = self.path_word(e1, e2)
                if not f or dist > 13:
                    continue

                self.candidate_pairs.append(tuple([tuple(e1), tuple(e2)]))
                self.candidate_pairs_dependency_dist.append([dist, dep_str])


    def generate_candidate_feature_vector(self, lexcion, test=False):
        if self.feature_vector != None:
            return
        i = 0
        self.feature_vector_dict = []
        for candidate_pair in self.candidate_pairs:
            self.feature_vector_dict.append(self.create_pair_feature_vector(lexcion, candidate_pair, i, test))
            i += 1

    def generate_feature_vector(self, lexcion):
        self.feature_vector = []
        word_len, pos_tag_len= len(lexcion["unigram"]["word"]), len(lexcion["unigram"]["pos_tag"])
        word_pos_tag_len = len(lexcion["unigram"]["word_pos_tag"])
        for f in self.feature_vector_dict:
            base = 0
            feat_vector = []

            word_index = [1,  #  特征词前一个词
                          2,  #  特征词前两个词
                          5,  #  特征词后一个词
                          6,  #  特征词后两个词
                          9,  #  情感词前一个词
                          10, #  情感词前两个词
                          13, #  情感词后一个词
                          14, #  情感词后两个词
                          17  #  特征词和情感词中间词
                         # 22 # 特征词
                         # 23 # 情感词
                          ]
            pos_tag_index = [3,  #  特征词前一个词 POS tag
                             4,  #  特征词前两个词 POS tag
                             7,  #  特征词后一个词 POS tag
                             8,  #  特征词后两个词 POS tag
                             11, #  情感词前一个词 POS tag
                             12, #  情感词前两个词 POS tag
                             15, #  情感词后一个词 POS tag
                             16, #  情感词后两个词 POS tag
                             18, #  特征词和情感词中间词 POS tag
                             20, #  特征词 POS tag
                             21  #  情感词 POS tag
                             ]
            word_pos_tag_index = [26, # 特征词前一个词以及POS tag
                                  27, # 特征词前两个词以及POS tag
                                  28, # 特征词后一个词以及POS tag
                                  29, # 特征词后两个词以及POS tag
                                  30, # 情感词前一个词以及POS tag
                                  31, # 情感词前两个词以及POS tag
                                  32, # 情感词后一个词以及POS tag
                                  33, # 情感词后两个词以及POS tag
                                  34  # 特征词和情感词中间词以及POS tag
                                  ]

            for i in word_index:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += word_len

            for i in pos_tag_index:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += pos_tag_len

            for i in word_pos_tag_index:
                for e in sorted(set(f[i])):
                    feat_vector.append(base + e)
                base += word_pos_tag_len

            # 特征词和情感词依赖树路径
            for e in sorted(set(f[19])):
                feat_vector.append(base + e)
            base += len(lexcion["unigram"]["dep"])

            # 依赖路径上词的个数
            for e in f[24]:
                feat_vector.append(base + e)
            base += 13

            # 两词之间是否有be
            for e in f[25]:
                feat_vector.append(base + e)
            base += 2

            # 特征词和情感词的相对顺序
            for e in f[0]:
                feat_vector.append(base + e)
            base += 2

            #  #  特征词
            #  for e in sorted(set(f[22])):
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  情感词
            #  for e in sorted(set(f[23])):
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词前一个词
            #  for e in f[1]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词前两个词
            #  for e in f[2]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词前一个词 POS tag
            #  for e in f[3]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  特征词前两个词 POS tag
            #  for e in f[4]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  特征词后一个词
            #  for e in f[5]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词后两个词
            #  for e in f[6]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词后一个词 POS tag
            #  for e in f[7]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  特征词后两个词 POS tag
            #  for e in f[8]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len



            #  #  情感词前一个词
            #  for e in f[9]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  情感词前两个词
            #  for e in f[10]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  情感词前一个词 POS tag
            #  for e in f[11]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  情感词前两个词 POS tag
            #  for e in f[12]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  情感词后一个词
            #  for e in f[13]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  情感词后两个词
            #  for e in f[14]:
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  情感词后一个词 POS tag
            #  for e in f[15]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  情感词后两个词 POS tag
            #  for e in f[16]:
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  特征词和情感词中间词
            #  for e in sorted(set(f[17])):
                #  feat_vector.append(base + e)
            #  base += word_len

            #  #  特征词和情感词中间词 POS tag
            #  for e in sorted(set(f[18])):
                #  feat_vector.append(base + e)
            #  base += pos_tag_len


            #  #  特征词 POS tag
            #  for e in sorted(set(f[20])):
                #  feat_vector.append(base + e)
            #  base += pos_tag_len

            #  #  情感词 POS tag
            #  for e in sorted(set(f[21])):
                #  feat_vector.append(base + e)
            #  base += pos_tag_len




            #  # 特征词前一个词以及POS tag
            #  for e in f[26]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 特征词前两个词以及POS tag
            #  for e in f[27]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 特征词后一个词以及POS tag
            #  for e in f[28]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 特征词后两个词以及POS tag
            #  for e in f[29]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 情感词前一个词以及POS tag
            #  for e in f[30]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 情感词前两个词以及POS tag
            #  for e in f[31]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 情感词后一个词以及POS tag
            #  for e in f[32]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len

            #  # 情感词后两个词以及POS tag
            #  for e in f[33]:
                #  feat_vector.append(base + e)
            #  base += word_pos_tag_len
            self.feature_vector.append(feat_vector)


    def create_pair_feature_vector(self, lexcion, pair, i_dep, test=False):
        ''''''
        # 特征词和情感词的相对顺序
        f0 = self.judge_feature_sentiment_order(pair[0], pair[1])

        # 特征词前一个词
        f1 = self.create_one_word(lexcion, pair[0][0]-1, f0[0], True, True, 0, test)
        # 特征词前两个词
        f2 = self.create_one_word(lexcion, pair[0][0]-2, f0[0], True, True, 0, test)
        # 特征词前一个词 POS tag
        f3 = self.create_one_word(lexcion, pair[0][0]-1, f0[0], True, True, 1, test)
        # 特征词前两个词 POS tag
        f4 = self.create_one_word(lexcion, pair[0][0]-2, f0[0], True, True, 1, test)
        # 特征词后一个词
        f5 = self.create_one_word(lexcion, pair[0][-1]+1, f0[0], True, False, 0, test)
        # 特征词后两个词
        f6 = self.create_one_word(lexcion, pair[0][-1]+2, f0[0], True, False, 0, test)
        # 特征词后一个词 POS tag
        f7 = self.create_one_word(lexcion, pair[0][-1]+1, f0[0], True, False, 1, test)
        # 特征词后两个词 POS tag
        f8 = self.create_one_word(lexcion, pair[0][-1]+2, f0[0], True, False, 1, test)


        # 情感词前一个词
        f9 = self.create_one_word(lexcion, pair[1][0]-1, f0[0], False, True, 0, test)
        # 情感词前两个词
        f10 = self.create_one_word(lexcion, pair[1][0]-2, f0[0], False, True, 0, test)
        # 情感词前一个词 POS tag
        f11 = self.create_one_word(lexcion, pair[1][0]-1, f0[0], False, True, 1, test)
        # 情感词前两个词 POS tag
        f12 = self.create_one_word(lexcion, pair[1][0]-2, f0[0], False, True, 1, test)
        # 情感词后一个词
        f13 = self.create_one_word(lexcion, pair[1][-1]+1, f0[0], False, False, 0, test)
        # 情感词后两个词
        f14 = self.create_one_word(lexcion, pair[1][-1]+2, f0[0], False, False, 0, test)
        # 情感词后一个词 POS tag
        f15 = self.create_one_word(lexcion, pair[1][-1]+1, f0[0], False, False, 1, test)
        # 情感词后两个词 POS tag
        f16 = self.create_one_word(lexcion, pair[1][-1]+2, f0[0], False, False, 1, test)



        words_list = self.get_between_word(pair[0], pair[1])
        # 特征词和情感词中间词
        f17 = self.create_words(lexcion, words_list, 0, test)
        # 特征词和情感词中间词 POS tag
        f18 = self.create_words(lexcion, words_list, 1, test)


        # 特征词和情感词依赖树路径
        f19 = self.create_dependency_path(lexcion, i_dep, test)

        # 特征词的 POS tag
        f20 = self.create_words(lexcion, pair[0], 1, test)

        # 情感词的 POS tag
        f21 = self.create_words(lexcion, pair[1], 1, test)

        # 特征词
        f22 = self.create_words(lexcion, pair[0], 0, test)

        # 情感词
        f23 = self.create_words(lexcion, pair[1], 0, test)

        # 依赖路径上词的个数
        f24 = self.create_dependency_word_count(lexcion, i_dep, test)

        # 两个词之间是否有 is
        f25 = self.judge_has_be(words_list)

        # 特征词前一个词以及POS tag
        f26 = self.create_one_word(lexcion, pair[0][0]-1, f0[0], True, True, 2, test)

        # 特征词前两个词以及POS tag
        f27 = self.create_one_word(lexcion, pair[0][0]-2, f0[0], True, True, 2, test)

        # 特征词后一个词以及POS tag
        f28 = self.create_one_word(lexcion, pair[0][-1]+1, f0[0], True, False, 2, test)

        # 特征词后两个词以及POS tag
        f29 = self.create_one_word(lexcion, pair[0][-1]+2, f0[0], True, False, 2, test)

        # 情感词前一个词以及POS tag
        f30 = self.create_one_word(lexcion, pair[1][0]-1, f0[0], True, True, 2, test)

        # 情感词前两个词以及POS tag
        f31 = self.create_one_word(lexcion, pair[1][0]-2, f0[0], True, True, 2, test)

        # 情感词后一个词以及POS tag
        f32 = self.create_one_word(lexcion, pair[1][-1]+1, f0[0], True, False, 2, test)

        # 情感词后两个词以及POS tag
        f33 = self.create_one_word(lexcion, pair[1][-1]+2, f0[0], True, False, 2, test)

        # 特征词和情感词中间词以及POS tag
        f34 = self.create_words(lexcion, words_list, 2, test)

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14, f15, f16, f17, f18, f19,
                f20, f21, f22, f23, f24, f25, f26, f27, f28,
                f29, f30, f31, f32, f33, f34]

    def judge_has_be(self, words_list):
        be_set = set(["is", "was", "are", "were", "am"])
        for i in words_list:
            if self.tokens[i].lower() in be_set:
                return [1]
        return [2]


    def create_dependency_word_count(self, lexcion, i_dep, test=False):
        f = []
        dist = self.candidate_pairs_dependency_dist[i_dep][0]
        f.append(dist)
        return f

    def create_dependency_path(self, lexcion, i_dep, test=False):
        f = []
        dep_list = self.candidate_pairs_dependency_dist[i_dep][1]
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
            if lexcion["unigram"]["dep"].get(e_low) == None:
                if test:
                    continue
                f.append(len(lexcion["unigram"]["dep"]) + 1)
                lexcion["unigram"]["dep"][e_low] = f[-1]
            else:
                f.append(lexcion["unigram"]["dep"][e_low])
        return f


    def judge_feature_sentiment_order(self, pair1, pair2):
        if pair1[-1] < pair2[0]:
            return [1]
        else:
            return [2]
    #  def create_words(self, lexcion, words_list, test=False):
        #  f = []
        #  for w in words_list:
            #  word_str = self.tokens[w].lower()
            #  if re.search(r"^\W+$", word_str) != None:
                #  continue
            #  if word_str in Static.stops:
                #  continue
            #  if lexcion["unigram"]["word"].get(word_str) == None:
                #  if test:
                    #  continue
                #  f.append(len(lexcion["unigram"]["word"]) + 1)
                #  lexcion["unigram"]["word"][word_str] = f[-1]
            #  else:
                #  f.append(lexcion["unigram"]["word"][word_str])
        #  return f

    #  def create_words_pos_tag(self, lexcion, words_list, test=False):
        #  f = []
        #  for w in words_list:
            #  word_str = self.pos_tag[w].lower()
            #  if re.search(r"^\W+$", word_str) != None:
                #  continue
            #  if lexcion["unigram"]["pos_tag"].get(word_str) == None:
                #  if test:
                    #  continue
                #  f.append(len(lexcion["unigram"]["pos_tag"]) + 1)
                #  lexcion["unigram"]["pos_tag"][word_str] = f[-1]
            #  else:
                #  f.append(lexcion["unigram"]["pos_tag"][word_str])
        #  return f
    def create_words(self, lexcion, words_list, option, test=False):
        '''
        @option: 0   word
                 1   pos_tag
                 2   word_pos_tag
        '''
        f = []
        for w in words_list:
            if option == 0:
                target = self.tokens[w].lower()
                lex = lexcion["unigram"]["word"]
                if target in Static.stops:
                    continue
            elif option == 1:
                target = self.pos_tag[w].lower()
                lex = lexcion["unigram"]["pos_tag"]
            else:
                target = " ".join([self.tokens[w].lower(), self.pos_tag[w].lower()])
                lex = lexcion["unigram"]["word_pos_tag"]

            if re.search(r"^\W+$", target) != None:
                continue
            if lex.get(target) == None:
                if test:
                    continue
                f.append(len(lex) + 1)
                lex[target] = f[-1]
            else:
                f.append(lex[target])
        return f

    #  def create_one_word(self, lexcion, k, order, is_feature, is_left, test=False):
        #  f = []
        #  if order == 1 and is_feature and not is_left:
            #  return f
        #  if order == 2 and is_feature and is_left:
            #  return f
        #  if order == 1 and not is_feature and is_left:
            #  return f
        #  if order == 2 and not is_feature and not is_left:
            #  return f
        #  word_str = "#" if self.tokens.get(k) == None else self.tokens[k].lower()
        #  if re.search(r"^\W+$", word_str) != None:
            #  return f
        #  if word_str in Static.stops:
            #  return f
        #  if lexcion["unigram"]["word"].get(word_str) == None:
            #  if test:
                #  return f
            #  f.append(len(lexcion["unigram"]["word"]) + 1)
            #  lexcion["unigram"]["word"][word_str] = f[0]
        #  else:
            #  f.append(lexcion["unigram"]["word"][word_str])
        #  return f

    #  def create_one_word_pos_tag(self, lexcion, k, order, is_feature, is_left, test=False):
        #  f = []
        #  if order == 1 and is_feature and not is_left:
            #  return f
        #  if order == 2 and is_feature and is_left:
            #  return f
        #  if order == 1 and not is_feature and is_left:
            #  return f
        #  if order == 2 and not is_feature and not is_left:
            #  return f
        #  pos_tag_str = "#" if self.tokens.get(k) == None else self.pos_tag[k].lower()
        #  if re.search(r"^\W+$", pos_tag_str) != None:
            #  return f
        #  if lexcion["unigram"]["pos_tag"].get(pos_tag_str) == None:
            #  if test:
                #  return f
            #  f.append(len(lexcion["unigram"]["pos_tag"]) + 1)
            #  lexcion["unigram"]["pos_tag"][pos_tag_str] = f[0]
        #  else:
            #  f.append(lexcion["unigram"]["pos_tag"][pos_tag_str])
        #  return f
    def create_one_word(self, lexcion, k, order, is_feature, is_left, option, test=False):
        '''
        @option: 0   word
                 1   pos_tag
                 2   word_pos_tag
        '''
        f = []
        if order == 1 and is_feature and not is_left:
            return f
        if order == 2 and is_feature and is_left:
            return f
        if order == 1 and not is_feature and is_left:
            return f
        if order == 2 and not is_feature and not is_left:
            return f
        if option == 0:
            target = "#" if self.tokens.get(k) == None else self.tokens[k].lower()
            lex = lexcion["unigram"]["word"]
            if target in Static.stops:
                return f
        elif option == 1:
            target = "#" if self.tokens.get(k) == None else self.pos_tag[k].lower()
            lex = lexcion["unigram"]["pos_tag"]
        else:
            target = "##" if self.tokens.get(k) == None else (" ".join([self.tokens[k].lower(), self.pos_tag[k].lower()]))
            lex = lexcion["unigram"]["word_pos_tag"]

        if re.search(r"^\W+$", target) != None:
            return f

        if lex.get(target) == None:
            if test:
                return f
            f.append(len(lex) + 1)
            lex[target] = f[0]
        else:
            f.append(lex[target])
        return f

    def generate_train_label(self):
        ''''''
        self.all_match_label = []
        for pair in self.candidate_pairs:
            label = 0
            for fs_ele in self.fs_dict.keys():
                if all_match(pair, fs_ele):
                    label = 1
                    break
            self.all_match_label.append(label)

        '''
        self.all_cover_label = []
        for pair in self.candidate_pairs:
            label = 0
            for fs_ele in self.fs_dict.keys():
                if all_cover(fs_ele, pair):
                    label = 1
                    break
            self.all_cover_label.append(label)

        self.have_part_label = []
        for pair in self.candidate_pairs:
            label = 0
            for fs_ele in self.fs_dict.keys():
                if have_part(pair, fs_ele):
                    label = 1
                    break
            self.have_part_label.append(label)
        '''

    def generate_test_label(self):
        self.all_match_label = [0 for e in self.candidate_pairs]


class SentenceTokenizer:

    # extract punctuation features from word list for position i
    # Features are: this word; previous word (lower case);
    # is the next word capitalized?; previous word only one char long?
    def punct_features(self, tokens, i):
        return {'next-word-capitalized': (i<len(tokens)-1) and (tokens[i+1][0].isupper()),
                'next-word': (i < len(tokens)-1) and re.search(r"^\w+$", tokens[i+1]),
                'prevword': tokens[i-1].lower(),
                'punct': tokens[i],
                'prev-word-is-one-char': len(tokens[i-1]) == 1}

    # The constructor builds a classifier using treebank training data
    # Naive Bayes is used for fast training
    # The entire dataset is used for training
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()

        self.word_pattern = re.compile(r"^([\w.]*)(\.)(\w*)$")
        self.proper_noun = re.compile(r"([A-Z]\.){2,}$")


        f = open("../../data/raw/transition_word.txt", "r", encoding="utf8")
        transition_word = f.readline()
        #print(transition_word)
        #words = r"([.,!?;:])\ *(and|but|or|however|because|so|therefore|also|since|thus|overall|including|especially|eventually|similarly)"
        self.words = r"([.,!?;:])\ *" + transition_word
        f.close()

        training_sents = nltk.corpus.treebank_raw.sents()
        tokens = []
        boundaries = set()
        offset = 0
        for sent in training_sents:
            tokens.extend(sent)
            offset += len(sent)
            boundaries.add(offset-1)

        # Create training features
        featuresets = [(self.punct_features(tokens,i), (i in boundaries))
                       for i in range(1, len(tokens)-1)
                       if tokens[i] in '.?!']

        train_set = featuresets
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Use the classifier to segment word tokens into sentences
    # words is a list of (word,bool) tuples
    def classify_segment_sentences(self,words):
        start = 0
        sents = []
        for i, word in enumerate(words):

            if word in '.?!' and self.classifier.classify(self.punct_features(words,i)) == True:
                sents.append(words[start:i+1])
                start = i+1
        if start < len(words):
            sents.append(words[start:])
        return sents

    def handle_word(self, text):
        text_words_sp = []
        for t in text:
            if re.search(r"\.{3,}", t):
                text_words_sp.append(t)
                continue
            if re.search(r"^[0-9]+\.[0-9]+", t):
                text_words_sp.append(t)
                continue
            if self.proper_noun.search(t):
                text_words_sp.append(t)
                continue
            obj = self.word_pattern.search(t)

            if obj:
                #print(t, obj.groups())
                if obj.groups()[0] != "":
                    if obj.groups()[2] != "":
                        text_words_sp.extend(obj.groups())
                    else:
                        text_words_sp.extend(obj.groups()[0:2])
                else:
                    text_words_sp.extend(obj.groups()[-2:])
            else:
                text_words_sp.extend(re.sub(r"\.", r"##########.", t).split("##########"))
        return text_words_sp

    # Segment text into sentences and words
    # returns a list of sentences, each sentence is a list of words
    # punctuation chars are classed as word tokens (except abbreviations)

    def nltk_check(self, sentences):
        sents = []
        for s in sentences:
            t = nltk.sent_tokenize(" ".join(s))
            for tt in t:
                ss = re.sub(r"^[^a-zA-Z(]*", "", tt)
                ss = re.sub(r"\(", "-LRB-", ss)
                ss = re.sub(r"\)", "-RRB-", ss)
                if re.search(r"^\W*$", ss):
                    continue
                sents.append(ss)

        return sents

    def split_junction(self, sentences):
        split_pattern = re.compile(self.words, re.I)
        sents = []
        for sentence in sentences:
            if len(sentence.split(" ")) < 20:
                sents.append(sentence)
                continue
            sent = re.split(r"##########", split_pattern.sub(r"\1##########\2", sentence))
            for w in sent:
                if len(w.split(" ")) >= 20:
                    w_list = re.split(r"\ [,;:]\ ", w)
                    sents.extend(w_list)
                else:
                    sents.append(w)
        return sents


    def segment_text(self,full_text):

        # Split (tokenize) text into words. Count whitespace as
        # words. Keeping this information allows us to distinguish between
        # abbreviations and sentence terminators
        text_words_sp_temp = self.tokenizer.tokenize(full_text)
        text_words_sp = self.handle_word(text_words_sp_temp)
        text_words_sp = [w for w in text_words_sp if w]

        sentences = self.classify_segment_sentences(text_words_sp)

        sentences = self.nltk_check(sentences)

        sentences = self.split_junction(sentences)

        return sentences



class Static:
    JJ = set(["JJ", "JJR", "JJS"])

    NN = set(["NN", "NNS", "NNP", "NNPS"])

    VB = set(["VB", "VBZ", "VBD", "VBG", "VBN", "VBP"])

    RB = set(["RB", "RBR", "RBS"])

    MR = set(load_file_line(r"../../data/raw/MR.txt"))

    BE = set(load_file_line(r"../../data/raw/BE.txt"))

    SENTIMENT = VB|RB|JJ

    terminal_signal = set(load_file_line(r"../../data/raw/terminal_signal.txt"))

    contrary = set(load_file_line(r"../../data/raw/contrary_word.txt"))

    #  weak_sentiment = set(load_file_line(r"../../data/raw/weak_sentiment.txt"))

    weak_sentiment = set(load_file_line(r"../../data/raw/weak_sentiment_add.txt"))

    positive_word = set(load_file_line(r"../../data/raw/positive-words.txt"))

    negative_word = set(load_file_line(r"../../data/raw/negative-words.txt"))

    weak_feature = set(load_file_line(r"../../data/raw/weak_feature.txt"))

    sentiment_word = {e:1 for e in positive_word}

    sentiment_word.update({e:-1 for e in negative_word})

    stops = set(stopwords.words('english'))


    def __init__(self):
        ''''''
        pass


class TrieNode:
    def __init__(self):
        self.count = 1
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add(self, sequence):
        node = self.root
        for c in sequence:
            if c not in node.children:
                child = TrieNode()
                node.children[c] = child
                node = child
            else:
                node = node.children[c]
                node.count += 1

    def count_seq(self, sequence):
        node = self.root
        for c in sequence:
            if c not in node.children:
                return 0
            else:
                node = node.children[c]
        return node.count



if __name__ == "__main__":
    #  t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
    #  # for s in t.subtrees(lambda t: t.label() in set(["NP", "VP"])):
    #  #    print(s.leaves())
    #  ret_dict = {}
    #  for i in range(len(t.leaves())):
        #  ret_dict[i+1] = []
        #  index = t.leaf_treeposition(i)
        #  n = len(index)
        #  for k in range(n-1, -1, -1):
            #  if k == n - 1 or t[index[:k]].label() == "VP":
                #  temp = t[index[:k]].leaves()
                #  if temp not in ret_dict[i+1]:
                    #  ret_dict[i+1].append(temp)

    #  print(ret_dict)
    #  print(Static.SENTIMENT)
    #  t = Trie()
    #  t.add(["how", "are", "how", "are"])
    #  t.add(["are", "how", "are"])
    #  t.add(["how", "are"])
    #  t.add(["are"])
    #  print(t.count_seq(["how"]))
    a = Sentence()
