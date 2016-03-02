# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
import sys, getopt
import pymysql

def usage():
    '''打印帮助信息'''
    print("create_database.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

def drop_table(connection, table_name):
    # 连接
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "drop table if exists " + table_name
            cursor.execute(sql)
        connection.commit()
    finally:
        pass

def create_database(name):
    # 连接
    connection = pymysql.connect(host="127.0.0.1",
                                user="root",
                                passwd="100704048",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "create database " + name
            cursor.execute(sql)
        connection.commit()
    finally:
        pass

def create_table_lm(connection, table_name, data_path):
    # 连接
    try:
        # 游标
        with connection.cursor() as cursor:
            #  sql = "create table if not exists {0} (id int unsigned not null auto_increment primary key, content varchar(255), score float(10, 5) not null)".format(table_name)
            sql = "create table if not exists {0} (id int unsigned not null auto_increment primary key, content varchar(255), score float(10, 5) not null) engine=myisam, data directory='{1}', index directory='{2}'".format(table_name, data_path, data_path)
            cursor.execute(sql)
        connection.commit()
    finally:
        pass

def create_table_num(connection, table_name, data_path):
    # 连接
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "create table if not exists {0} (i_pickle int unsigned not null, i_sentence int unsigned not null, num int unsigned not null, i_review int unsigned not null) engine=myisam, data directory='{1}', index directory='{2}'".format(table_name, data_path, data_path)
            cursor.execute(sql)
        connection.commit()
    finally:
        pass

def create_table_posting(connection, table_name, data_path):
    # 连接
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "create table if not exists {0} (i_pickle int unsigned not null, i_sentence int unsigned not null, b int unsigned not null, e int unsigned not null, i_content int unsigned not null) engine=myisam, data directory='{1}', index directory='{2}'".format(table_name, data_path, data_path)
            cursor.execute(sql)
        connection.commit()
    finally:
        pass

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:", ["help", "domain="])
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
    field_content = r"../../data/domains/" + content + r"/"
    connection = pymysql.connect(host="127.0.0.1",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    data_path = "/export/data/sharefolder/ecnucluster/u51141201057/datb1_12/database"
    #  drop_table(connection, content+"_lm")
    #  drop_table(connection, content+"_posting")
    #  drop_table(connection, content+"_num")
    #  create_table_lm(connection, content+"_lm", data_path)
    #  create_table_posting(connection, content+"_posting", data_path)
    create_table_num(connection, content+"_num", data_path)
    connection.close()
    print("end")
