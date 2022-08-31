# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/1 16:47

import pymysql
import pandas as pd
from sqlalchemy import create_engine


def create_db(db, table):
    connect = pymysql.connect(  # 连接数据库服务器
        user=db['user'],
        password=db['password'],
        host=db['host'],
        port=db['port'],
    )
    conn = connect.cursor()  # 创建操作游标
    # 你需要一个游标 来实现对数据库的操作相当于一条线索

    #                          创建表
    # conn.execute("drop database if exists " + db["db"])  # 如果new_database数据库存在则删除
    # conn.execute("create database " + db["db"])  # 新创建一个数据库
    conn.execute("use " + db["db"])  # 选择new_database这个数据库

    # sql 中的内容为创建一个名为new_table的表
    conn.execute("drop table if exists " + table)  # 如果表存在则删除
    # sql = """create table nlpccqa(id int primary key auto_increment,entity text character set utf8 collate utf8_unicode_ci,
    # attribute text character set utf8 collate utf8_unicode_ci, answer text character set utf8
    # collate utf8_unicode_ci,question text character set utf8 collate utf8_unicode_ci)auto_increment=0;"""  # ()中的参数可以自行设置
    # sql = "create table " + table + "(id int primary key auto_increment,qid text,entity text,relation text,answer text,question text)auto_increment=0;"
    sql = "create table " + table + "(id int primary key auto_increment,entity text,attribute text,answer text,question text)auto_increment=0;"
    conn.execute(sql)  # 创建表

    #                           删除
    # conn.execute("drop table new_table")

    conn.close()  # 关闭游标连接
    connect.close()  # 关闭数据库服务器连接 释放内存


def load_data(file, db, table):
    # 初始化数据库连接，使用pymysql模块
    db_info = {'user': db["user"],
               'password': db["password"],
               'host': db["host"],
               'port': db["port"],
               'database': db["db"]
               }
    # 采用了字典传值，字符串中接收字典中相同的名字key的value
    engine = create_engine(
        'mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8' % db_info, encoding='utf-8')
    # 直接使用下一种形式也可以
    # engine = create_engine('mysql+pymysql://root:123456@localhost:3306/test')

    # 读取本地CSV文件
    df = pd.read_csv(file, sep=',', encoding='utf-8')
    print(df)
    # 将新建的DataFrame储存为MySQL中的数据表，不储存index列(index=False)
    # if_exists:
    # 1.fail:如果表存在，啥也不做
    # 2.replace:如果表存在，删了表，再建立一个新表，把数据插入
    # 3.append:如果表存在，把数据插入，如果表不存在创建一个表！！
    pd.io.sql.to_sql(df, table, con=engine, index=False, if_exists='append', chunksize=10000)
    # df.to_sql('example', con=engine,  if_exists='replace')这种形式也可以
    print("Write to MySQL successfully!")


def search_data(sql):
    global results
    connect = pymysql.connect(  # 连接数据库服务器
        user="readonly",
        password="readonly",
        host="202.120.36.29",
        port=13306,
        db="zjp",
        charset="utf8"
    )
    cursor = connect.cursor()  # 创建操作游标
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
    except Exception as e:
        print("Error: unable to fecth data: %s ,%s" % (repr(e), sql))
    finally:
        # 关闭数据库连接
        cursor.close()
        connect.close()
    return results


if __name__ == '__main__':
    remote = {"user": "groupleader",
              "password": "onlyleaders",
              "host": "202.120.36.29",
              "port": 13306,
              "db": "zjp",
              "charset": "utf8"}
    local = {"user": "root",
             "password": "sdu.292653",
             "host": "127.0.0.1",
             "port": 3306,
             "db": "nlpccqa",
             "charset": "utf8"}
    # file = './nlpcc2016/clean_triple.csv'
    # file = './webqa/triples.csv'
    # table = "webqa"
    file = './originNLPCC2016ner/triple.csv'
    table = "originnlpccqa"
    create_db(remote, table)
    load_data(file, remote, table)
