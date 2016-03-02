# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
from my_package.class_define import Static
from my_package.scripts import load_pickle_file, create_content, save_pickle_file,\
    save_json_file
import sys, getopt
import re

def is_weak_feature(word):
    '''判断当前特征词 是否 weak
    '''
    weak_set = set(Static.weak_feature)
    for line in word.split():
        if line in weak_set:
            return True
    #  for line in Static.weak_feature:
        #  if line in set(word.split()):
            #  return True
    if re.search(r"^\W*$", word) != None:
        return True
    return False

def have_overlap(index1, index2):
    '''判断两个下标是否有重叠
    '''
    return bool(set(index1) & set(index2))


def judge_feature_word(sentence, score_pp_set, key, j=-1):
    #  判断找出特征词的词性是否满足要求，以及除去 BE
    if sentence.pos_tag[key] in Static.NN:
        pp = sentence.get_np(key, j)
    elif sentence.pos_tag[key] in Static.VB and sentence.tokens[key].lower() not in Static.BE:
        pp = sentence.get_vp(key, j)
    else:
        return None, False, None
    
    new_feature_string = sentence.get_phrase(pp).lower()
    #  判断找出特征词是否 weak
    if is_weak_feature(new_feature_string):
        return None, False, None
    
    #  判断给定特征词和找出情感词之间是否有交叉
    if have_overlap(pp, [j]):
        return None, False, None
    
    #  根据语言模型过滤某些词
    if new_feature_string not in score_pp_set:
        return None, False, None

    return pp, True, new_feature_string

def judge_sentiment_word(sentence, score_pp_set, key):
    #  判断找出情感词的词性是否满足
    if sentence.pos_tag[key] not in Static.SENTIMENT:
        return False, None
    
    #  去除 BE 动词
    if sentence.pos_tag[key] in Static.BE:
        return False, None
    
    #  判断找出情感词是否 weak
    new_sentiment_string = sentence.tokens[key].lower()
    if new_sentiment_string in Static.weak_sentiment:
        return False, None
    
    #  根据语言模型过滤某些词
    if new_sentiment_string not in score_pp_set:
        return False, None
    
    #  判断是否是　as well
    if sentence.judge_as_well(key):
        return False, None

    return True, new_sentiment_string



def s_using_s1(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c):
    '''根据情感词找出情感词的第一个规则
    '''
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        
        if sentence.sents_dict[tuple([key])] == None:
            continue
        polarity = sentence.sents_dict[tuple([key])]
        for value in values:
            
            #  判断依赖类型是否满足
            if value['type'] != "conj":
                continue
            

            
            #  #  判断找出情感词的词性是否满足
            #  if sentence.pos_tag[value['id']] not in Static.SENTIMENT:
                #  continue
            
            #  #  去除 BE 动词
            #  if sentence.pos_tag[value['id']] in Static.BE:
                #  continue
            
            #  #  判断找出情感词是否 weak
            #  new_sentiment_string = sentence.tokens[value['id']].lower()
            #  if new_sentiment_string in Static.weak_sentiment:
                #  continue
            
            #  #  根据语言模型过滤某些词
            #  if sentence.tokens[value['id']].lower() not in score_pp_set:
                #  continue
            
            #  #  判断是否是　as well
            #  if sentence.judge_as_well(value['id']):
                #  continue

            ret_f, new_sentiment_string = judge_sentiment_word(sentence, score_pp_set, value['id']) 
            if ret_f == False:
                continue


            #  判断找出的情感词是否之前已经被找出
            if  sentence.sents_dict[tuple([value['id']])] != None:
                continue

            #  判断找出情感词是否在sentiment_dict 中
            if sentiment_dict.get(new_sentiment_string) != None:
                continue
            

            #  赋予找出情感词的极性
            if sentence.check_not(key, value['id']) % 2 != 0:
                polarity = -1 * polarity
                
            # 加入新词词典，并根据极性相差个数决定是否加入　sentiment_dict
            if new_sentiment_word.get(new_sentiment_string) == None:
                new_sentiment_word[new_sentiment_string] = {}
                new_sentiment_word[new_sentiment_string][-1] = 0
                new_sentiment_word[new_sentiment_string][1] = 0
            new_sentiment_word[new_sentiment_string][polarity] += 1
            
            if new_sentiment_word[new_sentiment_string][1] - new_sentiment_word[new_sentiment_string][-1] >= c:
                polarity = 1
            elif new_sentiment_word[new_sentiment_string][-1] - new_sentiment_word[new_sentiment_string][1] >= c:
                polarity = -1
            else:
                continue
            sentiment_dict[new_sentiment_string] = polarity
           
            # name, single, 当前词, type1, up, (type2, down)
            regu = ("s_using_s1", True, sentence.tokens[value['id']],
                       value['type'], sentence.tokens[key], None, None)
            #添加元素到 sents, sents_regu, sents_set 中
            sentence.sents.append([[value['id']], polarity])
            sentence.sents_regu.append(regu)
            sentence.sents_dict[tuple([value['id']])] = polarity

def s_using_s2(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c):
    '''根据情感词找出情感词的第二个规则
    '''
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        for i_value in values:
            # 判断给定情感词是否出现在当前句子 feature_sentiment pair 中
            if sentence.sents_dict[tuple([i_value['id']])] == None:
                continue
            polarity = sentence.sents_dict[tuple([i_value['id']])]
            for j_value in values:
                if i_value == j_value:
                    continue
                #判断依赖类型是否相同
                if i_value['type'] != j_value['type']:
                    continue
    
                #  #判断找出情感词的词性是否满足
                #  if sentence.pos_tag[j_value['id']] not in Static.SENTIMENT:
                    #  continue
                
                #  #判断找出的情感词是否 weak
                #  new_sentiment_string = sentence.tokens[j_value['id']].lower()
                #  if new_sentiment_string in Static.weak_sentiment:
                    #  continue

                
                #  if sentence.judge_as_well(j_value['id']):
                    #  continue
                
                #  # 根据语言模型过滤某些词
                #  if sentence.tokens[j_value['id']].lower() not in score_pp_set:
                    #  continue

                #  #判断找出情感词是否在sentiment_dict 中
                #  if sentiment_dict.get(new_sentiment_string) != None:
                    #  continue
                
                ret_f, new_sentiment_string = judge_sentiment_word(sentence, score_pp_set, j_value['id']) 
                if ret_f == False:
                    continue

                #判断找出的情感词是否之前已经被找出
                if  sentence.sents_dict[tuple([j_value['id']])] != None:
                    continue
                
                #赋予找出的情感词的极性
                if sentence.check_not(i_value['id'], j_value['id']) % 2 != 0:
                    polarity = -1 * polarity

                # 加入新词词典，并根据极性相差个数决定是否加入　sentiment_dict
                if new_sentiment_word.get(new_sentiment_string) == None:
                    new_sentiment_word[new_sentiment_string] = {}
                    new_sentiment_word[new_sentiment_string][-1] = 0
                    new_sentiment_word[new_sentiment_string][1] = 0
                new_sentiment_word[new_sentiment_string][polarity] += 1
            
                if new_sentiment_word[new_sentiment_string][1] - new_sentiment_word[new_sentiment_string][-1] >= c:
                    polarity = 1
                elif new_sentiment_word[new_sentiment_string][-1] - new_sentiment_word[new_sentiment_string][1] >= c:
                    polarity = -1
                else:
                    continue

                # sentiment_dict[new_sentiment_string] = polarity
                
                # name, single, down1, type1, up, (type2, down2)
                regu = ("s_using_s2", False, sentence.tokens[i_value['id']],
                           i_value['type'], sentence.tokens[key], j_value['type'], 
                           sentence.tokens[j_value['id']])
                
                #添加元素到 sents, sents_regu, sents_set
                sentence.sents.append([[j_value['id']], polarity])
                sentence.sents_regu.append(regu)
                sentence.sents_dict[tuple([j_value['id']])] = polarity

def nsubj(sentence, sentiment_dict, score_pp_set, feature_set, key, value):
    '''当依赖类型是 "nsubj" 时特殊处理
    
    '''
    
    #ascertain feature and sentiment
    if sentence.pos_tag[key] in Static.NN:
        fea, sen = key, value['id']
        pp = sentence.get_np(fea, sen)
        
    elif sentence.pos_tag[key] in Static.VB and sentence.tokens[key].lower() not in Static.BE:
        fea, sen = key, value['id']
        pp = sentence.get_vp(fea, sen)
        
    elif sentence.pos_tag[value['id']] in Static.NN:
        fea, sen = value['id'], key
        pp = sentence.get_np(fea, sen)
    
    elif sentence.pos_tag[value['id']] in Static.VB and sentence.tokens[value['id']].lower() not in Static.BE:
        fea, sen = value['id'], key
        pp = sentence.get_vp(fea, sen)
    else:
        return
    
    given_sentiment_string = sentence.tokens[sen].lower()
    new_feature_string = sentence.get_phrase(pp).lower()
    
    #  判断是否是给定情感词
    if sentiment_dict.get(given_sentiment_string) == None:
        return
    
    #  判断给定情感词的词性是否满足
    if sentence.pos_tag[sen] not in Static.SENTIMENT:
        return

    #  判断是否是 as well
    if sentence.judge_as_well(sen):
        return

    #  去除 BE 动词
    if sentence.tokens[sen].lower() in Static.BE:
        return
    
    #  判断找出特征词是否 weak
    if is_weak_feature(new_feature_string):
        return
    
    #  判断找出特征词和给定情感词之间是否有交叉
    if have_overlap(pp, [sen]):
        return
    
    #  判断 fs pair 对是否已经找出
    if sentence.fs_dict.get((tuple(pp), tuple([sen]))) != None:
        return

    #语言模型过滤特征词
    if sentence.get_phrase(pp).lower() not in score_pp_set:
        return
    
    #赋予找出词对的极性
    polarity = sentiment_dict[given_sentiment_string]
    
    #feature_sentiment_ele -- (feature, sentiment, flag)
    #regu -- (name, single, down1, type1, up, type2, down2)
    feature_sentiment_ele = (pp, [sen], polarity)
    regu = ("f_using_s1(nsubj)", True, 
               sentence.tokens[fea],
               value['type'], sentence.tokens[sen], None, None)
    
    #加入 feature_sentiment, fs_regu, fs_set
    sentence.fs_add(feature_sentiment_ele, regu)
    sentence.fs_dict[(tuple(pp), tuple([sen]))] = polarity
    
    sentence.sents.append([[sen], polarity])
    sentence.sents_dict[tuple([sen])] = polarity
    sentence.sents_regu.append(regu)
    
    #加入 feats， feats_set， feats_regu
    sentence.feats.append(pp)
    sentence.feats_set.add(tuple(pp))
    sentence.feats_regu.append(regu)
    
    #加入到全局 feature_set
    feature_set.add(new_feature_string)

def f_using_s1(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set):
    '''根据情感词找出特征词，以及 feature-sentiment pair 的第一个规则 '''

    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        
        for value in values:
            if value['type'] not in Static.MR:
                continue
            #依赖类型为 "nsubj"时，特别处理
            if value['type'] == "nsubj":
                nsubj(sentence, sentiment_dict, score_pp_set, feature_set, key, value)
                continue
            
            #  判断是否是给定情感词
            given_sentiment_string = sentence.tokens[value['id']].lower()
            if sentiment_dict.get(given_sentiment_string) == None:
                continue
            
            #  判断给定情感词的词性是否满足
            if sentence.pos_tag[value['id']] not in Static.SENTIMENT:
                continue
            
            #  判断给定情感词是否是 as well 中的 well
            if sentence.judge_as_well(value['id']):
                continue
            
            #  判断找出特征词的词性是否满足要求，以及除去 BE
            if sentence.pos_tag[key] in Static.NN:
                pp = sentence.get_np(key, value['id'])
            elif sentence.pos_tag[key] in Static.VB and sentence.tokens[key].lower() not in Static.BE:
                pp = sentence.get_vp(key, value['id'])
            else:
                break
            
            new_feature_string = sentence.get_phrase(pp).lower()
            #  判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break
            
            #  判断给定特征词和找出情感词之间是否有交叉
            if have_overlap(pp, [value['id']]):
                continue

            #  根据语言模型过滤某些词
            if sentence.get_phrase(pp).lower() not in score_pp_set:
                continue
            
            #  判断 fs pair 对是否已经找出
            if sentence.fs_dict.get((tuple(pp), tuple([value['id']]))) != None:
                continue
            
            #  赋予找出词对的极性
            polarity = sentiment_dict[given_sentiment_string]
            
            #feature_sentiment_ele -- (feature, sentiment, flag)
            #regu -- (name, single, down1, type1, up, type2, down2)
            regu = ("f_using_s1", True, sentence.tokens[value['id']],
                       value['type'], sentence.tokens[key], None, None)
            feature_sentiment_ele = (pp, [value['id']], polarity)
            
            #  加入 feature_sentiment, fs_regu, fs_set
            sentence.fs_add(feature_sentiment_ele, regu)
            sentence.fs_dict[(tuple(pp), tuple([value['id']]))] = polarity
            
            sentence.sents.append([[value['id']], polarity])
            sentence.sents_dict[tuple([value['id']])] = polarity
            sentence.sents_regu.append(regu)
            
            #  加入 feats， feats_set， feats_regu
            sentence.feats.append(pp)
            sentence.feats_set.add(tuple(pp))
            sentence.feats_regu.append(regu)
            
            #  加入到全局 feature_set
            feature_set.add(new_feature_string)
            
def f_using_s2(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set):
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        if sentence.tokens[key].lower() in Static.weak_sentiment:
            continue
        for i_value in values:
            given_sentiment_string = sentence.tokens[i_value['id']].lower()
            #  判断给定情感词是否存在
            if sentiment_dict.get(given_sentiment_string) == None:
                continue
            
            #  判断给定情感词的词性是否满足要求
            if sentence.pos_tag[i_value['id']] not in Static.SENTIMENT:
                continue

            #  判断给定情感词是否是 as well 中的 well
            if sentence.judge_as_well(i_value['id']):
                continue
            
            for j_value in values:
                if i_value == j_value:
                    continue
                if i_value['type'] not in  Static.MR or j_value['type'] not in Static.MR:
                    continue

                #  #判断找出特征词的词性是否满足要求
                #  if sentence.pos_tag[j_value['id']] in Static.NN: 
                    #  pp = sentence.get_np(j_value['id'], i_value['id']) 
                #  elif sentence.pos_tag[j_value['id']] in Static.VB and sentence.tokens[j_value['id']].lower() not in Static.BE: 
                    #  pp = sentence.get_vp(j_value['id'], i_value['id'])
                #  else:
                    #  continue
                
                #  new_feature_string = sentence.get_phrase(pp).lower()
                #  #判断找出的特征词是否 weak
                #  if is_weak_feature(new_feature_string):
                    #  continue

                #  #判断特征词和情感词是否有交叉
                #  if have_overlap(pp, [i_value['id']]):
                    #  continue

                #  # 根据语言模型过滤某些词
                #  if sentence.get_phrase(pp).lower() not in score_pp_set:
                    #  continue

                pp, ret_f, new_feature_string = judge_feature_word(sentence, score_pp_set, j_value['id'], i_value['id'])
                if ret_f == False:
                    continue

                # 是否相邻
                if i_value['id'] - key != 1 and key - i_value['id'] != 1:
                    continue
                sent_list = sorted([i_value['id'], key])
                new_sentiment_string = sentence.get_phrase(sent_list).lower()
                
                #判断 fs pair 对是否已经找出
                if sentence.fs_dict.get((tuple(pp), tuple(sent_list))) != None:
                    continue
                
                if new_sentiment_string not in score_pp_set:
                    continue

                #赋予找出词对的情感词的极性
                polarity = sentiment_dict[given_sentiment_string]
                    
                #feature_sentiment_ele -- (feature, sentiment, flag)
                #regu -- (name, single, down1, type1, up, type2, down2)   
                feature_sentiment_ele = (pp, sent_list, polarity)
                regu = ("f_using_s2", False, sentence.tokens[i_value['id']],
                           i_value['type'], sentence.tokens[key], 
                           j_value['type'], sentence.tokens[j_value['id']])
                
                #加入 feature_sentiment, fs_regu, fs_set
                sentence.fs_add(feature_sentiment_ele, regu)
                sentence.fs_dict[(tuple(pp), tuple(sent_list))] = polarity
                
                sentence.sents.append([sent_list, polarity])
                sentence.sents_dict[tuple(sent_list)] = polarity
                sentence.sents_regu.append(regu)

                
                # 将两个词的情感词加入 sentiment_dict
                if sentiment_dict.get(new_sentiment_string) == None:
                    sentiment_dict[new_sentiment_string] = polarity
                
                #加入 feats， feats_set， feats_regu
                sentence.feats.append(pp)
                sentence.feats_set.add(tuple(pp))
                sentence.feats_regu.append(regu)
                
                #加入到全局 feature_set
                feature_set.add(new_feature_string)

def f_using_f1(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set):
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        
        #判断给定特征词是否在 fs pair 中出现过
        i = 0
        for e in sentence.feature_sentiment:
            if key in e[0]:
                given_fs_index = e
                break
            i += 1
        if i == len(sentence.feature_sentiment):
            continue
        
        #given_feature_string = sentence.get_phrase(given_feature_index).lower()
        for value in values:
            if value['type'] != "conj":
                continue

            #  #判断找出特征词的词性是否满足，以及不是 BE
            #  if sentence.pos_tag[value['id']] in Static.NN:
                #  pp = sentence.get_np(value['id'], given_fs_index[1][0])
            #  elif sentence.pos_tag[value['id']] in Static.VB and sentence.tokens[value['id']].lower() not in Static.BE:
                #  pp = sentence.get_vp(value['id'], given_fs_index[1][0])
            #  else:
                #  continue
            
            #  # 根据语言模型过滤某些词
            #  if sentence.get_phrase(pp).lower() not in score_pp_set:
                #  continue
            
            #  new_feature_string = sentence.get_phrase(pp).lower()
            #  #判断找出特征词是否 weak
            #  if is_weak_feature(new_feature_string):
                #  continue
            
            #  #判断找出fs pair 之间是否有重叠
            #  if have_overlap(feature_sentiment_ele[0], feature_sentiment_ele[1]):
                #  continue
            pp, ret_f, new_feature_string = judge_feature_word(sentence, score_pp_set, value['id'], given_fs_index[1][0])
            if ret_f == False:
                continue
            
            feature_sentiment_ele = (pp, given_fs_index[1], given_fs_index[2])
            #判断找出 fs pair 是否已经找过
            if sentence.fs_dict.get((tuple(feature_sentiment_ele[0]), tuple(feature_sentiment_ele[1]))) != None:
                continue
            
            regu = ("f_using_f1", True, sentence.tokens[value['id']],
                           value['type'], sentence.tokens[key], 
                           None, None)
            
            #加入 feature_sentiment, fs_regu, fs_set
            sentence.fs_add(feature_sentiment_ele, regu)
            sentence.fs_dict[(tuple(feature_sentiment_ele[0]), tuple(feature_sentiment_ele[1]))] = feature_sentiment_ele[2]
            
            #加入 feats， feats_set， feats_regu
            sentence.feats.append(pp)
            sentence.feats_set.add(tuple(pp))
            sentence.feats_regu.append(regu)
            
            #加入到全局 feature_set
            feature_set.add(new_feature_string)

def f_using_f2(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set):
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        for i_value in values:
            
            #判断给定特征词是否在 fs pair 中出现过
            i = 0
            for e in sentence.feature_sentiment:
                if i_value['id'] in e[0]:
                    given_fs_index = e
                    break
                i += 1
            if i == len(sentence.feature_sentiment):
                continue
            for j_value in values:
                if i_value == j_value:
                    continue
                if i_value['type'] != j_value['type']:
                    continue

                #  #判断找出特征词的词性是否满足，以及不是 BE
                #  if sentence.pos_tag[j_value['id']] in Static.NN:
                    #  pp = sentence.get_np(j_value['id'], given_fs_index[1][0])
                #  elif sentence.pos_tag[j_value['id']] in Static.VB and sentence.tokens[j_value['id']].lower() not in Static.BE:
                    #  pp = sentence.get_vp(j_value['id'], given_fs_index[1][0])
                #  else:
                    #  continue
                
                #  # 根据语言模型过滤某些词
                #  if sentence.get_phrase(pp).lower() not in score_pp_set:
                    #  continue
                #  new_feature_string = sentence.get_phrase(pp).lower()
                #  #判断找出特征词是否 weak
                #  if is_weak_feature(new_feature_string):
                    #  continue
                
                #  #判断找出fs pair 之间是否有重叠
                #  if have_overlap(feature_sentiment_ele[0], feature_sentiment_ele[1]):
                    #  continue

                pp, ret_f, new_feature_string = judge_feature_word(sentence, score_pp_set, j_value['id'], given_fs_index[1][0])
                if ret_f == False:
                    continue

                feature_sentiment_ele = (pp, given_fs_index[1], given_fs_index[2])
                #判断找出 fs pair 是否已经找过
                if sentence.fs_dict.get((tuple(feature_sentiment_ele[0]), tuple(feature_sentiment_ele[1]))) != None:
                    continue
                
                regu = ("f_using_f2", False, sentence.tokens[i_value['id']],
                               i_value['type'], sentence.tokens[key], 
                               j_value['type'], sentence.tokens[j_value['id']])
                
                #加入 feature_sentiment, fs_regu, fs_set
                sentence.fs_add(feature_sentiment_ele, regu)
                sentence.fs_dict[(tuple(feature_sentiment_ele[0]), tuple(feature_sentiment_ele[1]))] =feature_sentiment_ele[2]
                
                #加入 feats， feats_set， feats_regu
                sentence.feats.append(pp)
                sentence.feats_set.add(tuple(pp))
                sentence.feats_regu.append(regu)
                
                #加入到全局 feature_set
                feature_set.add(new_feature_string)
            
def s_using_f1(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c):
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        #判断给定特征词是否在本句中的 fs pair 中出现过
        if sentence.pos_tag[key] in Static.NN:
            pp = sentence.get_np(key)
        elif sentence.pos_tag[key] in Static.VB and sentence.tokens[key].lower() not in Static.BE:
            pp = sentence.get_vp(key)
        else:
            continue
        if tuple(pp) not in sentence.feats_set:
            continue
        for value in values:
            if value['type'] not in Static.MR:
                continue


            #  #判断找出情感词词性是否满足
            #  if sentence.pos_tag[value['id']] not in Static.SENTIMENT:
                #  continue
            #  #判断找出情感词是否weak
            #  if new_sentiment_string in Static.weak_sentiment:
                #  continue
            
            #  if have_overlap(pp, [value['id']]):
                #  continue
            
            #  if sentence.judge_as_well(value['id']):
                #  continue
            
            #  # 根据语言模型过滤某些词
            #  if sentence.tokens[value['id']].lower() not in score_pp_set:
                #  continue

            ret_f, new_sentiment_string = judge_sentiment_word(sentence, score_pp_set, value['id'])
            if ret_f == False:
                continue

            #判断该词是否已存在 sentiment_dict
            if sentiment_dict.get(new_sentiment_string) != None:
                continue

            if sentence.sents_dict[tuple([value['id']])] != None:
                continue
            
            #赋予找出情感词的极性
            polarity = sentence.polarity
            if sentence.check_not(key, value['id']) % 2 != 0:
                polarity = -1 * polarity
            
            # 加入新词词典，并根据极性相差个数决定是否加入　sentiment_dict
            if new_sentiment_word.get(new_sentiment_string) == None:
                new_sentiment_word[new_sentiment_string] = {}
                new_sentiment_word[new_sentiment_string][-1] = 0
                new_sentiment_word[new_sentiment_string][1] = 0
            new_sentiment_word[new_sentiment_string][polarity] += 1
            
            if new_sentiment_word[new_sentiment_string][1] - new_sentiment_word[new_sentiment_string][-1] >= c:
                polarity = 1
            elif new_sentiment_word[new_sentiment_string][-1] - new_sentiment_word[new_sentiment_string][1] >= c:
                polarity = -1
            else:
                continue

            #sentiment_dict[new_sentiment_string] = polarity
            #feature_sentiment_ele = [pp, [value['id']], polarity]                
            # name, single, 当前词, type1, up, (type2, down)
            regu = ("s_using_f1", True, sentence.tokens[value['id']],
                       value['type'], sentence.tokens[key], None, None)
            
            #sentence.fs_set.add((tuple(pp), tuple([value['id']])))
            #sentence.feature_sentiment.append(feature_sentiment_ele)
            #sentence.fs_regu.append(regu)
            
            #添加元素到 sents, sents_regu, sents_set 中
            sentence.sents.append([[value['id']], polarity])
            sentence.sents_regu.append(regu)
            sentence.sents_dict[tuple([value['id']])] = polarity
            
            
def s_using_f2(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c):
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        for i_value in values:
            
            for j_value in values:
                if i_value == j_value:
                    continue
                
                if i_value['type'] not in Static.MR or j_value['type'] not in Static.MR:
                        continue
                
                #判断给定特征词是否在本句中的 fs pair 中出现过
                if sentence.pos_tag[i_value['id']] in Static.NN:
                    pp = sentence.get_np(i_value['id'])
                elif sentence.pos_tag[i_value['id']] in Static.VB and sentence.tokens[i_value['id']].lower() not in Static.BE:
                    pp = sentence.get_vp(i_value['id'])
                else:
                    continue
                if tuple(pp) not in sentence.feats_set:
                    continue

                #  #判断找出情感词词性是否满足
                #  if sentence.pos_tag[j_value['id']] not in Static.SENTIMENT:
                    #  continue
                #  #判断找出情感词是否weak
                #  if new_sentiment_string in Static.weak_sentiment:
                    #  continue
                
                #  if sentence.judge_as_well(j_value['id']):
                    #  continue
                
                #  # 根据语言模型过滤某些词
                #  if sentence.tokens[j_value['id']].lower() not in score_pp_set:
                    #  continue

                ret_f, new_sentiment_string = judge_sentiment_word(sentence, score_pp_set, j_value['id'])
                if ret_f == False:
                    continue
                
                #判断该词是否已存在 sentiment_dict
                if sentiment_dict.get(new_sentiment_string) != None:
                    continue

                if sentence.sents_dict[tuple([j_value['id']])] != None:
                    continue

                #赋予找出情感词的极性
                polarity = sentence.polarity
                if sentence.check_not(key, j_value['id']) % 2 != 0:
                    polarity = -1 * polarity
                # 加入新词词典，并根据极性相差个数决定是否加入　sentiment_dict
                if new_sentiment_word.get(new_sentiment_string) == None:
                    new_sentiment_word[new_sentiment_string] = {}
                    new_sentiment_word[new_sentiment_string][-1] = 0
                    new_sentiment_word[new_sentiment_string][1] = 0
                new_sentiment_word[new_sentiment_string][polarity] += 1
                
                if new_sentiment_word[new_sentiment_string][1] - new_sentiment_word[new_sentiment_string][-1] >= c:
                    polarity = 1
                elif new_sentiment_word[new_sentiment_string][-1] - new_sentiment_word[new_sentiment_string][1] >= c:
                    polarity = -1
                else:
                    continue
             
                #sentiment_dict[new_sentiment_string] = polarity
                    
                #feature_sentiment_ele = [pp, [j_value['id']], polarity]
                # name, single, 当前词, type1, up, (type2, down)
                regu = ("s_using_f2", False, sentence.tokens[j_value['id']],
                           i_value['type'], sentence.tokens[key], j_value['type'], sentence.tokens[j_value['id']])
                
                #sentence.fs_set.add((tuple(pp), tuple([j_value['id']])))
                #sentence.feature_sentiment.append(feature_sentiment_ele)
                #sentence.fs_regu.append(regu)
                
                #添加元素到 sents, sents_regu, sents_set 中
                sentence.sents.append([[j_value['id']], polarity])
                sentence.sents_regu.append(regu)
                sentence.sents_dict[tuple([j_value['id']])] = polarity
                
            
def write_feature_sentiment(sentence, f, i):
    '''将句子中的 feature-sentiment pair 以及相应所用的规则输出
    '''
    #row -- (feature, sentiment, flag)
    #regular -- (name, single, down1, type1, up, type2, down2)
    #  for k in range(len(sentence.feature_sentiment)):
        
        #  print("{}:{}\n{} {}    {};   ".format(i,
                                 #  sentence.text,
                                 #  sentence.feature_sentiment[k],
                                 #  sentence.get_phrase(sentence.feature_sentiment[k][0]),
                                 #  sentence.get_phrase(sentence.feature_sentiment[k][1]),
                                 #  ), end="", file=f)
        #  print("{}:   {} -> {} -> {}".format(sentence.fs_regu[k][0],
                                     #  sentence.fs_regu[k][2],
                                     #  sentence.fs_regu[k][3],
                                     #  sentence.fs_regu[k][4]), end="", file=f)
        #  if sentence.fs_regu[k][1] == False:
            #  print(" <- {} <- {}".format(sentence.fs_regu[k][5], sentence.fs_regu[k][6]), end="", file=f)
        #  print("\n", file=f)
    print("{0}:{1}".format(i, sentence.text), file=f)
    for k in range(len(sentence.feature_sentiment)):
        
        print("{0}\t\t{1}".format(sentence.get_phrase(sentence.feature_sentiment[k][0]),
                                 sentence.get_phrase(sentence.feature_sentiment[k][1]),
                                 int(sentence.feature_sentiment[k][2] != 0)), file=f)
    print(file=f)

def write_feature(sentence, f, i):
    '''将句子中抽出特征词以及相应所用的规则输出
    '''
    for k in range(len(sentence.feats)):
        print("{}:{}\n{}\n{}: {}   {} -> {} -> {}".format(i,
                                     sentence.text_and_pos_tag,
                                     sentence.text, 
                                     sentence.get_phrase(sentence.feats[k]),
                                     sentence.feats_regu[k][0],
                                     sentence.feats_regu[k][2],
                                     sentence.feats_regu[k][3],
                                     sentence.feats_regu[k][4]), end="", file=f)
        if sentence.feats_regu[k][1] == False:
            print(" <- {} <- {}".format(sentence.feats_regu[k][5], sentence.feats_regu[k][6]), end="", file=f)
        print("\n", file=f)
        
def write_sentiment(sentence, f, i):
    '''将句子中抽出情感词以及相应所用的规则输出
    '''
    for k in range(len(sentence.sents)):
        print(sentence.sents[k], sentence.get_phrase(sentence.sents[k]), file=f) 
        print("{}:{}\n{}\n{} {}: {}   {} -> {} -> {}".format(i, 
                                     sentence.text_and_pos_tag,
                                     sentence.text,
                                     sentence.get_phrase(sentence.sents[k][0]),
                                     sentence.sents[k][1],
                                     sentence.sents_regu[k][0],
                                     sentence.sents_regu[k][2],
                                     sentence.sents_regu[k][3],
                                     sentence.sents_regu[k][4]), end="", file=f)
        if sentence.sents_regu[k][1] == False:
            print(" <- {} <- {}".format(sentence.sents_regu[k][5], sentence.sents_regu[k][6]), end="", file=f)
        print("\n", file=f)
            
def run(field_content, sentiment_dict, iter_num, b, e, c):
    ''' 运行该领域内的  bootstrap '''
    iter_count = 0
    score_pp_dict = load_pickle_file(field_content + r"pickles/score_pp.pickle")
    score_pp_set = set(score_pp_dict.keys())
    
    # 根据语言模型筛选通用情感词典
    sentiment_dict = {key : value for key, value in sentiment_dict.items() if key in score_pp_set}
    #  save_json_file(field_content + "pickles/raw_sentiment_dict.json", sentiment_dict)
    sent_dict = set(sentiment_dict.keys())
    print("sentiment dict len: ", len(sentiment_dict))
    create_content(field_content + r"bootstrap")
    create_content(field_content + r"pickles/bootstrap_sentences")
    new_sentiment_word, new_feature_word = {}, set() 
    feature_set = set()
    while iter_count < iter_num:
        print("iter :", iter_count+1)
        f1 = open(field_content + r"bootstrap/feature_sentiment_iter" + str(iter_count+1), "w", encoding="utf8")
        f2 = open(field_content + r"bootstrap/features_iter" + str(iter_count+1), "w", encoding="utf8")
        f3 = open(field_content + r"bootstrap/sentiments_iter" + str(iter_count+1), "w", encoding="utf8")
        i = b
        while i < e and os.path.exists(field_content + r"pickles/parse_sentences/parse_sentences_" + str(i) + ".pickle.bz2"):
            print("loading")
            if iter_count == 0:
                sentences = load_pickle_file(field_content + r"pickles/parse_sentences/parse_sentences_" + str(i) + ".pickle")
            else:
                sentences = load_pickle_file(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle")
            print("loaded")
            #  kkk = 0
            for sentence in sentences:
                sentence.set_init()
                #  if kkk != 25180:
                    #  kkk += 1
                    #  continue
                #  kkk += 1
                #  print(sentence.text_and_pos_tag)
                #  print(sentence.dependency_tree)
                f_using_s1(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set)
                f_using_s2(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set)
                f_using_f1(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set)
                f_using_f2(sentence, sentiment_dict, score_pp_set, new_feature_word, feature_set)
                s_using_f1(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c)
                s_using_f2(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c)
                s_using_s2(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c)
                s_using_s1(sentence, sentiment_dict, score_pp_set, new_sentiment_word, feature_set, c)
            for k in range(len(sentences)):
                if len(sentences[k].feature_sentiment) == 0:
                    continue
                #  write_feature_sentiment(sentences[k], f1, (i-1)*60000 + 1+k)
                #  write_feature(sentences[k], f2, (i-1)*60000 + 1+k)
                #  write_sentiment(sentences[k], f3, (i-1)*60000 + 1+k)
                write_feature_sentiment(sentences[k], f1, k)
                write_feature(sentences[k], f2, k)
                write_sentiment(sentences[k], f3, k)
            save_pickle_file(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle", sentences)
            if iter_count == iter_num - 1:
                save_sentences = [sentence for sentence in sentences if sentence.feature_sentiment != []]
                save_pickle_file(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle", save_sentences)
            i += 1
        f1.close()
        f2.close()
        f3.close()
        iter_count += 1
    new_dict = {key : value for key, value in sentiment_dict.items() if key not in sent_dict} 
    save_json_file(field_content + r"pickles/new_sent_dict.json", new_dict)
    save_json_file(field_content + r"pickles/new_sentiment_word.json", new_sentiment_word)
    save_pickle_file(field_content + r"pickles/bootstrap_sentiment_dict.pickle", sentiment_dict)
    save_json_file(field_content + r"pickles/bootstrap_sentiment_dict.json", sentiment_dict)
    return sentiment_dict

def usage():
    '''打印帮助信息'''
    print("bootstrap_regular.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-b, --begin: parse_sentences pickel 文件的开始编号(包含此文件)")
    print("-e, --end: parse_sentences pickel 文件的结束编号(不包含此文件)")
    print("-i, --iter: boostrap 的迭代次数")
    print("-c, --count: 同一个词正负词性出现次数差值的绝对值")
        
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:i:c:", ["help", "domain=", "begin=", "end=", "iter=", "count="])
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
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)
        if op in ("-i", "--iter"):
            iter_num = int(value)
        if op in ("-c", "--count"):
            c = int(value)

    field_content = r"../../data/domains/" + content + r"/"
    run(field_content, dict(Static.sentiment_word), iter_num, b, e, c)
