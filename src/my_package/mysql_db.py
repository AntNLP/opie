#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/05 19:26:16

@author: Changzhi Sun
"""
import configparser
import os

import pymysql


class MysqlDB:

    def __init__(self, domain):
        cf = configparser.ConfigParser()
        cf.read(os.path.join(os.getenv("OPIE_DIR"), "my.cnf"))
        self.conf = dict(cf.items("mysql"))
        self.preprocess()
        self.conn = pymysql.connect(host=self.conf["host"],
                                    user=self.conf["user"],
                                    passwd=self.conf["passwd"],
                                    charset=self.conf["charset"],
                                    cursorclass=pymysql.cursors.DictCursor)
        self.create_db(self.conf["db"])
        self.conn.close()
        self.conn = pymysql.connect(host=self.conf["host"],
                                    user=self.conf["user"],
                                    passwd=self.conf["passwd"],
                                    db=self.conf["db"],
                                    charset=self.conf["charset"],
                                    cursorclass=pymysql.cursors.DictCursor)
        self.domain = domain

    def __del__(self):
        self.conn.close()

    def preprocess(self):
        pass

    def execute(self, sql):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
        finally:
            pass

    def create_db(self, db):
        sql = "create database if not exists " + db
        self.execute(sql)

    def create_lm(self):
        sql = "create table if not exists {0} ".format(self.domain + "_lm")
        sql += "(id int unsigned not null auto_increment primary key, "
        sql += "content varchar(255), score float(10, 5) not null) "
        sql += "engine=myisam, "
        sql += "data directory='{0}', ".format(self.conf["data_index_path"])
        sql += "index directory='{0}'".format(self.conf["data_index_path"])
        self.execute(sql)

    def create_posting(self):
        sql = "create table if not exists {0}".format(self.domain + "_posting")
        sql += " (i_pickle int unsigned not null, "
        sql += "i_sentence int unsigned not null, b int unsigned not null, "
        sql += "e int unsigned not null, i_content int unsigned not null) "
        sql += "engine=myisam, "
        sql += "data directory='{0}', ".format(self.conf["data_index_path"])
        sql += "index directory='{0}'".format(self.conf["data_index_path"])
        self.execute(sql)

    def create_gopwd(self):
        '''general opinion word infomation for each sentence'''

        sql = "create table if not exists {0} ".format(self.domain + "_gopwd")
        sql += "(i_pickle int unsigned not null, "
        sql += "i_sentence int unsigned not null, "
        sql += "pos_num int unsigned not null, "
        sql += "neg_num int unsigned not null, "
        sql += "i_review int unsigned not null) engine=myisam, "
        sql += "data directory='{0}', ".format(self.conf["data_index_path"])
        sql += "index directory='{0}'".format(self.conf["data_index_path"])
        self.execute(sql)

    def drop_table(self, table):
        sql = "drop table if exists " + table
        self.execute(sql)

    def drop_db(self, table):
        sql = "drop table if exists " + table
        self.execute(sql)

    def language_model(self, word, threshold=-25):
        try:
            with self.connection.cursor() as cursor:
                sql = "select * from %s " % (self.domain + "_lm")
                sql += "where content=\"%s\" " % word
                sql += "and score>=%d" % threshold
                cursor.execute(sql)
                res = cursor.fetchall()
                if len(res) == 0:
                    return False
                else:
                    return True
        except Exception as err:
            return False
        finally:
            pass

if __name__ == "__main__":
    mydb = Database("cell")
    mydb.create_lm()
    mydb.create_posting()
    mydb.create_gopwd()
    print("end")
