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
    for line in word.split():
        if line in Static.weak_feature:
            return True
    if re.search(r"^\W*$", word) != None:
        return True
    return False

def have_overlap(index1, index2):
    '''判断两个下标是否有重叠
    '''
    return bool(set(index1) & set(index2))

def nn_mod_and_obj(sentence, features, sentiments, score_pp_set, new_feature_string, tuple_list, key, value, r_list, is_swap=False):
    if sentence.dependency_tree.get(key) != None:
        for i_value in sentence.dependency_tree[key]:
            if i_value['type'] not in r_list:
                continue
            if sentence.pos_tag[i_value['id']] not in Static.NN:
                continue
            np = sentence.get_np(i_value['id'], key)
            new_np_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_np_string):
                break
         
            # 语言模型过滤
            if new_np_string not in score_pp_set:
                break
         
            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value, key]):
                continue

            features.add(new_np_string)
            if is_swap:
                new_sentiment_string = sentence.tokens[value].lower() +" " + sentence.tokens[key].lower()
            else:
                new_sentiment_string = sentence.tokens[key].lower() +" " + sentence.tokens[value].lower()
            tuple_list.append([new_np_string, new_sentiment_string])
            return False
    else:
        return True
    return True


def nn_npadvmod(sentence, features, sentiments, score_pp_set, new_feature_string, tuple_list, key, value):

    return nn_mod_and_obj(sentence, features, sentiments, score_pp_set, 
            new_feature_string, tuple_list, key, value, ["npadvmod"], True)

def nn_dobj(sentence, features, sentiments, score_pp_set, new_feature_string, tuple_list, key, value):
    return nn_mod_and_obj(sentence, features, sentiments, score_pp_set, 
            new_feature_string, tuple_list, key, value, ["dobj"])

def sent_weak_lang_overlap(new_sentiment_string, score_pp_set, pp, qq):
    # 判断新找出的情感词时候weak
    if new_sentiment_string in Static.weak_sentiment:
        return True
        
    # 语言模型过滤
    if new_sentiment_string not in score_pp_set:
        return True

    # 判断特征词和找出情感词是否有重叠
    if have_overlap(pp, qq):
        return True

    return False

def jj_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, value, pp):
    if sentence.dependency_tree.get(value) == None:
        return
    for sent_value in sentence.dependency_tree[value]:
        if sent_value['type'] not in ["conj"]:
            continue
        if sentence.pos_tag[sent_value['id']] not in Static.JJ:
            continue
        new_sentiment_string = sentence.tokens[sent_value['id']].lower()
        if sent_weak_lang_overlap(new_sentiment_string, score_pp_set, pp, [sent_value['id']]):
            continue
        sentiments.add(new_sentiment_string)
        tuple_list.append([new_feature_string, new_sentiment_string])

def nn_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, key, pp):
    if sentence.dependency_tree_up.get(key) != None:
        up = sentence.dependency_tree_up[key]
    else:
        return 
    if sentence.pos_tag[up] not in Static.JJ:
        return 
    for i_value in sentence.dependency_tree[up]:
        if i_value['id'] != key:
            continue
        if i_value['type'] not in["conj"]:
            continue
        new_sentiment_string = sentence.tokens[up].lower()

        # 判断新找出的情感词时候weak
        if new_sentiment_string in Static.weak_sentiment:
            continue
        # 语言模型过滤
        if new_sentiment_string not in score_pp_set:
            continue

        # 判断特征词和找出情感词是否有重叠
        if have_overlap(pp, [up]):
            continue
        if sent_weak_lang_overlap(new_sentiment_string, score_pp_set, pp, [up]):
            continue
        tuple_list.append([new_feature_string, new_sentiment_string])
        return


def f_using_s1(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        for value in values:
            if value['type'] not in ["amod", "dep"]:
                continue
            
            given_sentiment_string = sentence.tokens[value['id']].lower()

            # 判断给定情感词在不在 sentiments 
            if given_sentiment_string not in sentiments:
                continue

            # 判断给定情感词的词性
            if sentence.pos_tag[value['id']] not in Static.JJ:
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.NN:
                break

            np = sentence.get_np(key, value['id'])
            new_feature_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value['id']]):
                continue

            features.add(new_feature_string)
            tuple_list.append([new_feature_string, given_sentiment_string])
            if value['type'] == "amod":
                jj_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, value['id'], np)
            else:
                nn_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, key, np)


            
    return tuple_list

def f_using_s2(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        for value in values:
            if value['type'] not in ["xcomp", "acomp"]:
                continue
            
            given_sentiment_string = sentence.tokens[value['id']].lower()

            # 判断给定情感词在不在 sentiments 
            if given_sentiment_string not in sentiments:
                continue

            # 判断给定情感词的词性
            if sentence.pos_tag[value['id']] not in Static.JJ:
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.VB:
                break

            vp = sentence.get_vp(key, value['id'])
            new_feature_string = sentence.get_phrase(vp).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            # 判断特征词和情感词是否有重叠
            if have_overlap(vp, [value['id']]):
                continue

            features.add(new_feature_string)
            tuple_list.append([new_feature_string, given_sentiment_string])
            jj_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, value['id'], vp)

    return tuple_list

def f_using_s3(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        given_sentiment_string = sentence.tokens[key].lower()

        # 判断给定情感词在不在 sentiments 
        if given_sentiment_string not in sentiments:
            continue
  
        # 判断给定情感词的词性
        if sentence.pos_tag[key] not in Static.JJ:
            continue

        for value in values:
            if value['type'] not in ["nsubj"]:
                continue
            
            # 判断找出特征词的词性
            if sentence.pos_tag[value['id']] not in Static.NN:
                continue

            np = sentence.get_np(value['id'], key)
            new_feature_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [key]):
                continue

            features.add(new_feature_string)
            tuple_list.append([new_feature_string, given_sentiment_string])
            jj_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, key, np)
    return tuple_list
            
def f_using_s4(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        for value in values:
            if value['type'] not in ["advmod"]:
                continue
            
            given_sentiment_string = sentence.tokens[value['id']].lower()
            # 判断给定情感词在不在 sentiments 
            if given_sentiment_string not in sentiments:
                continue

            # 判断给定情感词的词性
            if sentence.pos_tag[value['id']] not in Static.RB:
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.VB:
                break

            new_feature_string = sentence.tokens[key].lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            add_f1 = nn_dobj(sentence, features, sentiments, score_pp_set,
                    new_feature_string, tuple_list, key, value['id'])
            add_f2 = nn_npadvmod(sentence, features, sentiments, score_pp_set,
                    new_feature_string, tuple_list, value['id'], key)
            features.add(new_feature_string)
            tuple_list.append([new_feature_string, given_sentiment_string])

    return tuple_list

def f_using_s5(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        for value in values:
            if value['type'] not in ["advmod"]:
                continue
            
            given_sentiment_string = sentence.tokens[value['id']].lower()

            # 判断给定情感词在不在 sentiments 
            if given_sentiment_string not in sentiments:
                continue

            # 判断给定情感词的词性
            if sentence.pos_tag[value['id']] not in Static.RB:
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.NN:
                break

            np = sentence.get_np(key, value['id'])
            new_feature_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value['id']]):
                continue

            features.add(new_feature_string)
            tuple_list.append([new_feature_string, given_sentiment_string])
            jj_conj_jj(sentence, sentiments, score_pp_set, new_feature_string, tuple_list, value['id'], np)

    return tuple_list

def f_using_s6(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        # 判断情感词的词性
        if sentence.pos_tag[key] not in Static.VB:
            continue

        # 判断情感词是否weak
        if sentence.tokens[key].lower() in Static.weak_sentiment:
            continue

        for value in values:
            if value['type'] not in ["xcomp"]:
                continue
            
            # 判断找出情感词的词性
            if sentence.pos_tag[value['id']] not in Static.VB:
                continue

            #  判断情感词是否weak
            #  if sentence.tokens[value['id']].lower() in Static.weak_sentiment:
                #  continue

            if sentence.dependency_tree.get(value['id']) == None:
                continue

            i_np, i_to = None, None
            for i_value in sentence.dependency_tree[value['id']]:
                if i_value['type'] in ["dobj"] and sentence.pos_tag[i_value['id']] in Static.NN:
                    np = sentence.get_np(i_value['id'], value['id'])
                    i_np = i_value['id']
                if i_value['type'] in ['aux']:
                    i_to = i_value['id']
                
            if i_np == None or i_to == None:
                continue
            new_feature_string = sentence.get_phrase(np).lower()
            new_sentiment_string = sentence.tokens[key].lower() + " " + sentence.tokens[i_to].lower() + " " + sentence.tokens[value['id']].lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                continue

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                continue
            if new_sentiment_string not in score_pp_set:
                continue

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [key, i_to, value['id']]):
                continue


            features.add(new_feature_string)
            tuple_list.append([new_feature_string, new_sentiment_string])

    return tuple_list

def f_using_s7(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue

        # 判断给定情感词的词性
        if sentence.pos_tag[key] not in Static.NN:
            continue

        for value in values:
            if value['type'] not in ["nsubj"]:
                continue
            
            # 判断特征词的词性
            if sentence.pos_tag[value['id']] not in Static.NN:
                continue
            np1 = sentence.get_np(value['id'], key)
            new_feature_string = sentence.get_phrase(np1).lower()

            np = sentence.get_np(key, value['id'])

            if len(np) == 1:
                continue
            f_x = False
            for x in np[:-1]:
                if sentence.pos_tag[x] in Static.NN:
                    f_x = True
                if sentence.pos_tag[x] in Static.JJ:
                    f_x = True
                if sentence.pos_tag[x] in Static.RB:
                    f_x = True
                if sentence.pos_tag[x] in Static.VB:
                    f_x = True

            if not f_x:
                break

            new_sentiment_string = sentence.get_phrase(np).lower()
            # 判断找出情感词是否 weak
            if is_weak_feature(new_sentiment_string):
                break

            # 语言模型过滤
            if new_sentiment_string not in score_pp_set:
                break

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, np1):
                continue

            features.add(new_feature_string)
            sentiments.add(new_sentiment_string)
            tuple_list.append([new_feature_string, new_sentiment_string])
    return tuple_list

def f_using_s8(sentence, features, sentiments, score_pp_set):
    tuple_list = []
    for key, values in sentence.dependency_tree.items():
        if values == None or key == 0:
            continue
        for value in values:
            if value['type'] not in ["rcmod"]:
                continue
            
            #  given_sentiment_string = sentence.tokens[value['id']].lower()
            #  # 判断给定情感词在不在 sentiments 
            #  if given_sentiment_string not in sentiments:
                #  continue

            # 判断给定情感词的词性
            if sentence.pos_tag[value['id']] not in Static.VB:
                continue
            if sentence.tokens[value['id']] in Static.weak_sentiment:
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.NN:
                break

            np = sentence.get_np(key, value['id'])
            new_feature_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 语言模型过滤
            if new_feature_string not in score_pp_set:
                break

            if sentence.dependency_tree.get(value['id']) == None:
                continue
            
            for i_value in sentence.dependency_tree[value['id']]:
                if i_value['type'] not in ["advmod"]:
                    continue
                if sentence.pos_tag[i_value['id']] not in Static.RB:
                    continue
                new_sentiment_string = sentence.tokens[i_value['id']].lower() + " " + sentence.tokens[value['id']].lower()
                if new_feature_string not in score_pp_set:
                    continue
                features.add(new_feature_string)
                tuple_list.append([new_feature_string, new_sentiment_string])
    return tuple_list

def run(field_content, sentiment_dict):
    ''' 运行该领域内的  bootstrap '''
    print("loading...")
    sentences = load_pickle_file(field_content+r"pickles/parse_sentences/parse_sentences_1.pickle")
    print("loaded...")

    score_pp_dict = load_pickle_file(field_content + r"pickles/score_pp.pickle")
    score_pp_set = set(score_pp_dict.keys())

    sentiment_dict = {key : value for key, value in sentiment_dict.items() if key in score_pp_set}
    sentiments = set(sentiment_dict.keys())
    features = set()

    f = open(field_content + "bootstrap/result", "w", encoding="utf8")
    k = 0
    for sentence in sentences:
        tuple_list = []
        #  if k != 10127:
            #  k += 1
            #  continue
        #  print(sentence.text_and_pos_tag)
        #  print(sentence.dependency_tree)
        tuple_list = f_using_s1(sentence, features, sentiments, score_pp_set)
        tuple_list.extend(f_using_s2(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s3(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s4(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s5(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s6(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s7(sentence, features, sentiments, score_pp_set))
        tuple_list.extend(f_using_s8(sentence, features, sentiments, score_pp_set))
        if tuple_list != []:
            #  print("{0}:{1}\n{2}".format(k, sentence.text_and_pos_tag, sentence.text), file=f)
            print("{0}:{1}".format(k, sentence.text), file=f)
            for w1, w2 in tuple_list:
                print("{0}\t\t{1}".format(w1, w2), file=f)
            print(file=f)
        k += 1
    f.close()
    with open(field_content+"pickles/sentiments", "w") as out:
        for sent in sentiments:
            print(sent, file=out)

    with open(field_content+"pickles/features", "w") as out:
        for feat in features:
            print(feat, file=out)


        
if __name__ == "__main__":
    field_content = r"../../data/domains/reviews_Cell_Phones_and_Accessories/"
    run(field_content, dict(Static.sentiment_word))

