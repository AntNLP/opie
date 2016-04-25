# -*- coding: utf-8 -*-
'''
Created on 2015年8月28日

@author: Changzhi Sun
'''
import pickle
import os
import json
import bz2
from json.encoder import JSONEncoder


def remove(filename):
    if os.path.exists(filename):
        os.remove(filename)


def load_json_file(filename):
    ''' load json 文件

    keyword argument:

    file -- 文件路径

    return:
    相应的变量

    '''
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(filename, var):
    '''将变量 var 序列化到 file 文件

    keyword argument:

    file -- 保存的文件路径
    var -- 需要保存的变量

    '''
    with open(filename, mode="w", encoding="utf8") as f:
        json.dump(var, f, indent=2, cls=ClsEncoder)


def mkdir(dirname):
    '''若当前目录不存在，则创建当前目录
    '''
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def load_pickle_file(filename):
    ''' load pickle 文件

    keyword argument:

    file -- 文件路径

    return:
    相应的变量

    '''
    if not filename.endswith(".bz2"):
        if os.path.exists(filename):
            with open(filename, "rb") as out:
                return pickle.load(out)
        else:
            filename = filename + ".bz2"
            with bz2.open(filename, "rb") as out:
                return pickle.load(out)
    else:
        with bz2.open(filename, "rb") as out:
            return pickle.load(out)


def save_pickle_file(filename, var):
    '''将变量 var 序列化到 file 文件

    keyword argument:

    file -- 保存的文件路径
    var -- 需要保存的变量

    '''
    if not filename.endswith(".bz2"):
        filename = filename + ".bz2"
    with bz2.open(filename, "wb") as out:
        pickle.dump(var, out)


def load_file_line(filename):
    '''load 文件

    keyword argument:

    file -- 文件路径

    return:
    去除每行换行以后的序列

    '''
    with open(filename, mode="r", encoding="utf-8") as out:
        for line in out:
            strip_line = line.strip()
            if not strip_line.startswith("#"):
                yield strip_line


def read_weak(file_name):
    ''''''
    with open(file_name, "r", encoding="utf8") as out:
        index = 1
        for line in out:
            line = line.strip()
            entry = {}
            for l in line.split(" "):
                s = l.split("=")
                if (len(s) == 1):
                    entry['stemmed1'] += " " + l
                else:
                    entry[s[0]] = s[1]
            index += 1
            if entry['type'] == "weaksubj":
                yield entry['word1']


def create_weak_file(file_name):
    ''''''
    with open(r"../../data/raw/weak_sentiment_raw.txt", "w", encoding="utf8") as out:
        for e in read_weak(file_name):
            print(e, file=out)


def read_to_set(file_name):
    ''''''
    with open(file_name, mode="r", encoding="utf-8") as out:
        word_set = set()
        for line in out:
            word_set.add(line.strip())
        return word_set


def del_common():
    ''''''
    weak_file1 = r"../../data/raw/weak_sentiment_raw.txt"
    weak_file2 = r"../../data/raw/weak_sentiment_add.txt"
    sen_file1 = r"../../data/raw/negative-words.txt"
    sen_file2 = r"../../data/raw/positive-words.txt"
    set1 = read_to_set(weak_file1)
    set2 = read_to_set(weak_file2)
    set3 = read_to_set(sen_file1)
    set4 = read_to_set(sen_file2)
    with open(r"../../data/raw/common.txt", "w", encoding="utf-8") as out:
        for e in (set1 | set2) & (set3 | set4):
            print(e, file=out)

    common_file = r"../../data/raw/common.txt"
    weak_file = r"../../data/raw/weak_sentiment_raw.txt"
    fina_file = r"../../data/raw/weak_sentiment.txt"
    set1 = read_to_set(common_file)
    set2 = read_to_set(weak_file)
    with open(fina_file, "w", encoding="utf-8") as out:
        for e in (set2 - set1):
            print(e, file=out)


def get_index(connection, table_lm, var):
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "select * from {0} where content=\"{1}\"".format(table_lm, var)
            cursor.execute(sql)
            res = cursor.fetchall()
            if len(res) == 0:
                return None
            else:
                return res[0]['id']

    except Exception as err:
        print(err)
        print(var)
        return None
    finally:
        pass


def get_position(connection, table_posting, var):
    try:

        # 游标
        with connection.cursor() as cursor:
            sql = "select distinct i_pickle, i_sentence from {0} where i_content={1}".format(table_posting, var)
            cursor.execute(sql)
            res = cursor.fetchall()
            return res
    except Exception as err:
        print(err)
        return None
    finally:
        pass

if __name__ == "__main__":
    create_weak_file(r"../../data/raw/subjclueslen1-HLTEMNLP05.tff")
    del_common()
