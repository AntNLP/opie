# -*- coding: utf-8 -*-
'''
Created on 2015年8月28日

@author: Changzhi Sun
'''
import pickle
import os
import json
from json.encoder import JSONEncoder
import bz2

def all_match(pair1, pair2):
    '''判断两个 pair 是否完全匹配

    keyword argument:

    pair1 -- 当前一个 pair
    pair2 -- 当前另一个 pair

    return:

    如果完全匹配，返回 True，否则返回 False

    '''
    return pair1 == pair2

def all_cover(pair1, pair2):
    '''判断两个 pair2 是否完全覆盖 pair1

    keyword argument:

    pair1 -- 当前一个 pair
    pair2 -- 当前另一个 pair

    return:

    如果完全覆盖，返回 True，否则返回 False

    '''
    set1 = set(pair1[0])
    set2 = set(pair1[1])
    set3 = set(pair2[0])
    set4 = set(pair2[1])
    if set1 & set3 == set1 and set2 & set4 == set2:
        return True
    return False

def have_part(pair1, pair2):
    '''判断 pair1 和 pair2 是否有重叠的部分

    keyword argument:

    pair1 -- 当前一个 pair
    pair2 -- 当前另一个 pair

    return:

    如果对应部分都有重叠，则返回 True，否则返回 False

    '''
    set1 = set(pair1[0])
    set2 = set(pair1[1])
    set3 = set(pair2[0])
    set4 = set(pair2[1])
    if (set1 & set3 != set()) and (set2 & set4 != set()):
        return True
    return False


def obj2dict(obj):

    #if isinstance(obj, bytes):
    #    return {'__class__': 'bytes',
    #            '__value__': list(obj)}
    memberlist = [m for m in dir(obj)]
    _dict = {}
    for m in memberlist:
        if m[0] != '_' and not callable(m):
            _dict[m] = getattr(obj, m)
    return _dict
    raise TypeError(repr(object) + ' is not JSON serializable')


class ClsEncoder(JSONEncoder):
    def default(self, o):
        return obj2dict(o)

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

def create_content(content_name):
    '''若当前目录不存在，则创建当前目录
    '''
    if not os.path.exists(content_name):
            os.mkdir(content_name)
    else:
        print("目录已经存在！" + content_name)

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

def return_none():
    return None

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




if __name__ == "__main__":
    create_weak_file(r"../../data/raw/subjclueslen1-HLTEMNLP05.tff")
    del_common()
