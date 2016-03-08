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
import pymysql

def inquire_content(connection, var, table_lm, t=-25):
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

def is_as_well(sentence, k):
    if sentence.tokens[k].lower() == "well" and k > 1 and sentence.tokens[k-1].lower() == "as":
        return True
    return False

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


def nn_mod_and_obj(sentence, features, sentiments, new_feature_string, key, value, r_list, connection, table_lm, regu_name, is_swap=False):
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

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value, key]):
                continue

            # 语言模型过滤
            if not inquire_content(connection, new_np_string, table_lm):
                break


            features.add(new_np_string)
            if is_swap:
                # 判断之前是否已经找出
                if sentence.fs_dict.get((tuple(np), tuple([value, key]))) != None:
                    continue
                new_sentiment_string = sentence.tokens[value].lower() +" " + sentence.tokens[key].lower()
                #  sentiments.add(new_sentiment_string)
                sentence.fs_dict[(tuple(np), tuple([value, key]))] = 1
                sentence.feature_sentiment.append([np, [value, key]])
                sentence.fs_regu.append(regu_name)
            else:
                # 判断之前是否已经找出
                if sentence.fs_dict.get((tuple(np), tuple([key, value]))) != None:
                    continue
                new_sentiment_string = sentence.tokens[key].lower() +" " + sentence.tokens[value].lower()

                # 语言模型过滤
                if not inquire_content(connection, new_sentiment_string, table_lm):
                    break
                #  sentiments.add(new_sentiment_string)
                sentence.fs_dict[(tuple(np), tuple([key, value]))] = 1
                sentence.feature_sentiment.append([np, [key, value]])
                sentence.fs_regu.append(regu_name)
            return False
    else:
        return True
    return True

def nn_dobj(sentence, features, sentiments, new_feature_string, key, value, connection, table_lm, regu_name):
    return nn_mod_and_obj(sentence, features, sentiments, 
            new_feature_string, key, value, ["dobj"], connection, table_lm, regu_name)

def sent_weak_lang_overlap(new_sentiment_string, pp, qq, connection, table_lm):
    # 判断新找出的情感词时候weak
    if new_sentiment_string in Static.weak_sentiment:
        return True

    # 判断特征词和找出情感词是否有重叠
    if have_overlap(pp, qq):
        return True

    # 语言模型过滤
    if not inquire_content(connection, new_sentiment_string, table_lm):
        return True

    return False

def jj_conj_jj(sentence, sentiments, new_feature_string, value, pp, connection, table_lm, regu_name, add_complex=False):
    if sentence.dependency_tree.get(value) == None:
        return
    for sent_value in sentence.dependency_tree[value]:
        if sent_value['type'] in ["conj"] and sentence.pos_tag[sent_value['id']] in Static.JJ:
            i_jj = sent_value['id']
            new_sentiment_string = sentence.tokens[i_jj].lower()
            if sent_weak_lang_overlap(new_sentiment_string, pp, [i_jj], connection, table_lm):
                continue
            # 判断之前是否已经找出
            if sentence.fs_dict.get((tuple(pp), tuple([i_jj]))) != None:
                continue
            #  sentiments.add(new_sentiment_string)
            sentence.fs_dict[(tuple(pp), tuple([i_jj]))] = 1
            sentence.feature_sentiment.append([pp, [i_jj]])
            sentence.fs_regu.append(regu_name)
            if not add_complex:
                continue
            for i_value in sentence.dependency_tree[value]:
                if i_value['type'] in ['xcomp'] and sentence.pos_tag[i_value['id']] in Static.VB:
                    i_vb = i_value['id']
                    if sentence.dependency_tree.get(i_vb) != None:
                        i_to = None
                        for v in sentence.dependency_tree[i_vb]:
                            if v['type'] in ['aux']:
                                i_to = v['id']
                                break
                        if i_to == None:
                            continue
                        if i_to != i_jj + 1 or i_to != i_vb - 1:
                            continue
                        new_sentiment_string = sentence.tokens[i_jj].lower() + " " + sentence.tokens[i_to].lower() + " " + sentence.tokens[i_vb].lower()
                        if new_sentiment_string in Static.weak_sentiment:
                            continue

                        # 判断特征词和找出情感词是否有重叠
                        if have_overlap(pp, [i_jj, i_to, i_vb]):
                            continue

                        # 判断之前是否已经找出
                        if sentence.fs_dict.get((tuple(pp), tuple([i_jj, i_to, i_vb]))) != None:
                            continue

                        # 语言模型过滤
                        if not inquire_content(connection, new_sentiment_string, table_lm):
                            continue

                        #  print(new_sentiment_string)
                        #  sentiments.add(new_sentiment_string)
                        sentence.fs_dict[(tuple(pp), tuple([i_jj, i_to, i_vb]))] = 1
                        sentence.feature_sentiment.append([pp, [i_jj, i_to, i_vb]])
                        sentence.fs_regu.append(regu_name)

def nn_conj_jj(sentence, sentiments, new_feature_string, key, pp, connection, table_lm, regu_name):
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

        # 判断特征词和找出情感词是否有重叠
        if have_overlap(pp, [up]):
            continue
        if sent_weak_lang_overlap(new_sentiment_string, pp, [up], connection, table_lm):
            continue

        # 判断之前是否已经找出
        if sentence.fs_dict.get((tuple(pp), tuple([up]))) != None:
            continue

        #  语言模型过滤
        if not inquire_content(connection, new_sentiment_string, table_lm):
            continue
        sentence.fs_dict[(tuple(pp), tuple([up]))] = 1
        sentence.feature_sentiment.append([pp, [up]])
        sentence.fs_regu.append(regu_name)
        return

def jj_to_vp(sentence, sentiments, sent, new_feature_string, i_feat, connection, table_lm, regu_name):
    if sent not in sentence.dependency_tree:
        return
    for value in sentence.dependency_tree[sent]:
        if value['type'] not in ["xcomp"]:
            continue
        if sentence.pos_tag[value['id']] not in Static.VB:
            continue
        i_vb = value['id']
        if i_vb not in sentence.dependency_tree:
            continue
        i_to = None
        for v in sentence.dependency_tree[i_vb]:
            if v['type'] in ['aux']:
                i_to = v['id']
                break
        if i_to == None:
            continue

        # 三个词连续
        if i_to != sent + 1 or i_to != i_vb - 1:
            continue

        # 判断特征词和情感词是否有重叠
        if have_overlap([sent, i_to, i_vb], i_feat):
            continue

        # 判断之前是否已经找出
        if sentence.fs_dict.get((tuple(i_feat), tuple([sent, i_to, i_vb]))) != None:
            continue
        new_sentiment_string = " ".join([sentence.tokens[sent].lower(), sentence.tokens[i_to].lower(), sentence.tokens[i_vb].lower()])

        # 语言模型过滤
        if not inquire_content(connection, new_sentiment_string, table_lm):
            continue

        #  sentiments.add(new_sentiment_string)
        sentence.fs_dict[(tuple(i_feat), tuple([sent, i_to, i_vb]))] = 1
        sentence.feature_sentiment.append([i_feat, [sent, i_to, i_vb]])
        sentence.fs_regu.append(regu_name)


def f_using_s1(sentence, features, sentiments, connection, table_lm):
    regu_name = "f_using_s1"
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

            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value['id']]):
                continue

            # 判断之前是否已经找出
            if sentence.fs_dict.get((tuple(np), tuple([value['id']]))) != None:
                continue

            # 语言模型过滤
            if not inquire_content(connection, new_feature_string, table_lm):
                break
            features.add(new_feature_string)
            sentence.fs_dict[(tuple(np), tuple([value['id']]))] = 1
            sentence.feature_sentiment.append([np, [value['id']]])
            sentence.fs_regu.append(regu_name)
            jj_to_vp(sentence, sentiments, value['id'], new_feature_string, np, connection, table_lm, regu_name)
            if value['type'] == "amod":
                jj_conj_jj(sentence, sentiments, new_feature_string, value['id'], np, connection, table_lm, regu_name)
            else:
                nn_conj_jj(sentence, sentiments, new_feature_string, key, np, connection, table_lm, regu_name)

def f_using_s2(sentence, features, sentiments, connection, table_lm):
    regu_name = "f_using_s2"
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


            # 判断特征词和情感词是否有重叠
            if have_overlap(vp, [value['id']]):
                continue

            # 判断之前是否已经找出
            if sentence.fs_dict.get((tuple(vp), tuple([value['id']]))) != None:
                continue

            # 语言模型过滤
            if not inquire_content(connection, new_feature_string, table_lm):
                break
            features.add(new_feature_string)
            sentence.fs_dict[(tuple(vp), tuple([value['id']]))] = 1
            sentence.feature_sentiment.append([vp, [value['id']]])
            sentence.fs_regu.append(regu_name)
            jj_to_vp(sentence, sentiments, value['id'], new_feature_string, vp, connection, table_lm, regu_name)
            jj_conj_jj(sentence, sentiments, new_feature_string, value['id'], vp, connection, table_lm, regu_name)

def f_using_s3(sentence, features, sentiments, connection, table_lm):
    regu_name = "f_using_s3"
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

            # 判断是否是 as well
            if is_as_well(sentence, value['id']):
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.VB:
                break

            new_feature_string = sentence.tokens[key].lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break

            # 判断之前是否已经找出
            if sentence.fs_dict.get((tuple([key]), tuple([value['id']]))) != None:
                continue

            # 语言模型过滤
            if not inquire_content(connection, new_feature_string, table_lm):
                break

            sentence.fs_dict[(tuple([key]), tuple([value['id']]))] = 1
            features.add(new_feature_string)
            sentence.feature_sentiment.append([[key], [value['id']]])
            sentence.fs_regu.append(regu_name)
            nn_dobj(sentence, features, sentiments, 
                    new_feature_string, key, value['id'], connection, table_lm, regu_name)


def f_using_s4(sentence, features, sentiments, connection, table_lm):
    regu_name = "f_using_s4"
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

            # 判断是否是 as well
            if is_as_well(sentence, value['id']):
                continue

            # 判断找出特征词的词性
            if sentence.pos_tag[key] not in Static.NN:
                break

            np = sentence.get_np(key, value['id'])
            new_feature_string = sentence.get_phrase(np).lower()

            # 判断找出特征词是否 weak
            if is_weak_feature(new_feature_string):
                break


            # 判断特征词和情感词是否有重叠
            if have_overlap(np, [value['id']]):
                continue

            # 判断之前是否已经找出
            if sentence.fs_dict.get((tuple(np), tuple([value['id']]))) != None:
                continue

            # 语言模型过滤
            if not inquire_content(connection, new_feature_string, table_lm):
                break
            features.add(new_feature_string)
            sentence.fs_dict[(tuple(np), tuple([value['id']]))] = 1
            sentence.feature_sentiment.append([np, [value['id']]])
            sentence.fs_regu.append(regu_name)
            jj_conj_jj(sentence, sentiments, new_feature_string, value['id'], np, connection, table_lm, regu_name)

def have_dependent(dependency_tree, pp, sp):
    sp_set = set(sp)
    pp_set = set(pp)
    for e in pp:
        if e not in dependency_tree:
            continue
        for value in dependency_tree[e]:
            if value['id'] in sp_set:
                return True
    for e in sp:
        if e not in dependency_tree:
            continue
        for value in dependency_tree[e]:
            if value['id'] in pp_set:
                return True
    return False


def f_using_s5(sentence, features, sentiments, connection, sentiment_dict, table_lm):
    regu_name = "f_using_s5"
    for i in range(1, len(sentence.pos_tag) + 1):
        if sentence.tokens[i] == None:
            continue
        if sentence.tokens[i].lower() not in ["is", "was", "are", "were", "am"]:
            continue
        j = i - 1
        pp = None
        while j > 0:
            if sentence.pos_tag[j] in ["PRP", "EX"]:
                break
            if sentence.pos_tag[j] in Static.NN:
                pp = sentence.get_np(j, i)
                break
            j -= 1
        if pp == None:
            continue
        new_feature_string = sentence.get_phrase(pp).lower()

        # 判断找出特征词是否 weak
        if is_weak_feature(new_feature_string):
            continue

        # 语言模型过滤
        if not inquire_content(connection, new_feature_string, table_lm):
            continue

        j = i + 1
        sp_nn, sp_vb, sp_jj = None, None, None
        while sentence.pos_tag.get(j) != None:
            if sentence.pos_tag[j] in Static.NN and sp_nn == None:
                sp_nn = sentence.get_np(j, i)
                flg = True
                for x in sp_nn:
                    if sentence.tokens[x].lower() in sentiment_dict:
                        flg = False
                        break
                if flg:
                    sp_nn = None
            elif sentence.pos_tag[j] in Static.VB and sp_vb == None:
                sp_vb = sentence.get_vp(j, i)
            elif sentence.pos_tag[j] in Static.JJ and sp_jj == None:
                sp_jj = sentence.get_max_adjp(j, [i])
            j += 1

        min_i = [e for e in [sp_nn, sp_vb, sp_jj] if e != None]
        if min_i == []:
            continue
        sp = min_i[0]
        for e in min_i:
            if e[0] < sp[0]:
                sp = e
        if sp == sp_vb and len(sp) == 1:
            continue
        new_sentiment_string= sentence.get_phrase(sp).lower()

        if new_sentiment_string in Static.weak_sentiment:
            continue

        if have_overlap(pp, sp):
            continue

        if not have_dependent(sentence.dependency_tree, pp, sp):
            continue

        # 判断之前是否已经找出
        if sentence.fs_dict.get((tuple(pp), tuple(sp))) != None:
            continue

        # 语言模型过滤
        if not inquire_content(connection, new_sentiment_string, table_lm):
            continue
        #  sentiments.add(new_sentiment_string)
        features.add(new_feature_string)
        sentence.fs_dict[(tuple(pp), tuple(sp))] = 1
        sentence.feature_sentiment.append([pp, sp])
        sentence.fs_regu.append(regu_name)


def write_feature_sentiment(sentence, f):
    '''将句子中的 feature-sentiment pair 以及相应所用的规则输出
    '''
    print("S\t{0}".format(sentence.text), file=f)
    for k in range(len(sentence.feature_sentiment)):
        key, value = sentence.feature_sentiment[k][0], sentence.feature_sentiment[k][1]
        print("R\t{0}\t{1}\t{2}\t{3}\t{4}".format(
            sentence.get_phrase(key).lower(),
            sentence.get_phrase(value).lower(),
            key,
            value,
            sentence.fs_regu[k]), file=f)

def run(field_content, sentiment_dict, b, e, connection, table_lm):
    ''' 运行该领域内的  bootstrap '''
    iter_count = 0

    # 根据语言模型筛选通用情感词典
    new_sentiment_dict = {key : value for key, value in sentiment_dict.items() if inquire_content(connection, key, table_lm)}
    sentiments = set(new_sentiment_dict.keys())
    #  sentiments = load_pickle_file(field_content + r"pickles/sentiments.pickle")
    features = set()
    #  features = load_pickle_file(field_content + r"pickles/features.pickle")
    create_content(field_content + r"bootstrap")
    create_content(field_content + r"pickles/bootstrap_sentences")
    f1 = open(field_content + r"bootstrap/feat_sent", "w", encoding="utf8")
    f2 = open(field_content + r"bootstrap/feat_iter", "w", encoding="utf8")
    f3 = open(field_content + r"bootstrap/sent_iter", "w", encoding="utf8")
    i = b
    while i < e and os.path.exists(field_content + r"pickles/parse_sentences/parse_sentences_" + str(i) + ".pickle.bz2"):
        print(i, "loading")
        sentences = load_pickle_file(field_content + r"pickles/parse_sentences/parse_sentences_" + str(i) + ".pickle")
        print(i, "loaded")
        for sentence in sentences:
            sentence.set_init()
            f_using_s1(sentence, features, sentiments, connection, table_lm)
            f_using_s2(sentence, features, sentiments, connection, table_lm)
            f_using_s3(sentence, features, sentiments, connection, table_lm)
            f_using_s4(sentence, features, sentiments, connection, table_lm)
            f_using_s5(sentence, features, sentiments, connection, sentiment_dict, table_lm)
        save_sentences = [sentence for sentence in sentences if sentence.feature_sentiment != []]
        for k in range(len(save_sentences)):
            write_feature_sentiment(save_sentences[k], f1)
        save_pickle_file(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle", save_sentences)
        i += 1
    for e in sentiments:
        print(e, file=f3)
    for e in features:
        print(e, file=f2)
    f1.close()
    f2.close()
    f3.close()
    iter_count += 1
    save_pickle_file(field_content + r"pickles/sentiments.pickle", sentiments)
    save_pickle_file(field_content + r"pickles/features.pickle", features)
    return sentiment_dict

def usage():
    '''打印帮助信息'''
    print("bootstrap_regular.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-b, --begin: parse_sentences pickel 文件的开始编号(包含此文件)")
    print("-e, --end: parse_sentences pickel 文件的结束编号(不包含此文件)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:", ["help", "domain=", "begin=", "end="])
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

    #  field_content = r"../../data/domains/" + content + r"/"
    field_content = r"../../data/soft_domains/" + content + r"/"
    table_lm = content + "_lm"
    connection = pymysql.connect(host="console",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    run(field_content, dict(Static.sentiment_word), b, e, connection, table_lm)
    connection.close()
