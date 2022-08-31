# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/1 16:47

"""
构造三元组训练集，存入数据库，用于KBQA数据库检索
"""

import pandas as pd

question_str = "<question"
triple_str = "<triple"
answer_str = "<answer"
start_str = "============="

all_triple_list = []
all_clean_triple_list = []

for data_type in ["training", "testing"]:
    triple_list = []
    clean_triple_list = []
    error_triple_list = []

    file = "./nlpcc2016/nlpcc-iccpol-2016.kbqa." + data_type + "-data"
    with open(file, 'r', encoding='utf-8') as f:
        q_str = ""
        t_str = ""
        a_str = ""
        for line in f:
            if question_str in line:
                q_str = line.strip()
            if triple_str in line:
                t_str = line.strip()
            if start_str in line:  # new question answer triple
                entities = t_str.split("|||")[0].strip().split(">")[1].strip()
                q_str = q_str.split(">")[1].strip()
                triples = t_str.split(">")[1].strip().split("|||")
                triple = []
                for item in triples:
                    triple.append(item.strip())
                triple.append(q_str)
                triple_list.append(triple)
                if entities in q_str:
                    clean_triples = t_str.split(">")[1].strip().split("|||")
                    clean_triple = []
                    for item in clean_triples:
                        clean_triple.append(item.strip())
                    clean_triple.append(q_str)
                    clean_triple_list.append(clean_triple)
                else:
                    error_triple_list.append(triple)
    all_triple_list.extend(triple_list)
    all_clean_triple_list.extend(clean_triple_list)
    # print(len(triple_list))
    # print(len(clean_triple_list))
    df = pd.DataFrame(triple_list, columns=["entity", "attribute", "answer", "question"])
    df.to_csv("./nlpcc2016/" + data_type + "_triple.csv", encoding='utf-8', index=False)

    df1 = pd.DataFrame(error_triple_list, columns=["entity", "attribute", "answer", "question"])
    df1.to_csv("./nlpcc2016/" + data_type + "_error_triple.csv", encoding='utf-8', index=False)

    df2 = pd.DataFrame(clean_triple_list, columns=["entity", "attribute", "answer", "question"])
    df2.to_csv("./nlpcc2016/" + data_type + "_clean_triple.csv", encoding='utf-8', index=False)

df3 = pd.DataFrame(all_triple_list, columns=["entity", "attribute", "answer", "question"])
df3.to_csv("./nlpcc2016/triple.csv", encoding='utf-8', index=False)

df4 = pd.DataFrame(all_clean_triple_list, columns=["entity", "attribute", "answer", "question"])
df4.to_csv("./nlpcc2016/clean_triple.csv", encoding='utf-8', index=False)
