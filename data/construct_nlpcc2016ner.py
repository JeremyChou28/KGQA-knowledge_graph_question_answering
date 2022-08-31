# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/1 16:47

"""
构造NER数据集，将序列进行实体标注，用于BERT+BiLSTM+CRF模型的NER训练
"""

import sys
import os
import pandas as pd

for data_type in ["train", "test"]:
    if data_type == "train":
        file = "./nlpcc2016/training_clean_triple.csv"
    elif data_type == "test":
        file = "./nlpcc2016/testing_clean_triple.csv"
    else:
        raise "error:the data_type should be train or test"

    question_str = "<question"
    triple_str = "<triple"
    answer_str = "<answer"
    start_str = "============="

    q_t_a_list = []
    seq_q_list = []
    seq_tag_list = []

    df = pd.read_csv(file)
    q_str = ""
    t_str = ""
    a_str = ""

    for i in range(len(df)):
        q_str = df.loc[i]['question'].strip()
        entities = str(df.loc[i]['entity']).strip()
        if entities in q_str:
            q_list = list(q_str)
            seq_q_list.extend(q_list)
            seq_q_list.extend([" "])
            tag_list = ["O" for i in range(len(q_list))]
            tag_start_index = q_str.find(entities)
            for i in range(tag_start_index, tag_start_index + len(entities)):
                if tag_start_index == i:
                    tag_list[i] = "B-LOC"
                else:
                    tag_list[i] = "I-LOC"
            seq_tag_list.extend(tag_list)
            seq_tag_list.extend([" "])
        else:
            pass
    # print('\t'.join(seq_tag_list[0:50]))
    # print('\t'.join(seq_q_list[0:50]))
    # print(len(seq_tag_list))
    # print(len(seq_q_list))
    seq_result = [str(q) + " " + tag for q, tag in zip(seq_q_list, seq_tag_list)]
    print(len(seq_result))
    if data_type == "test":  # 170533
        with open("./nlpcc2016ner/" + "test" + ".txt", "w", encoding='utf-8') as f:
            f.write("\n".join(seq_result[0:164900]))
            f.close()
        with open("./nlpcc2016ner/" + "dev" + ".txt", "r+", encoding='utf-8') as f:
            f.write("\n".join(seq_result[164900:]))
            f.close()
    elif data_type == "train":  # 241692
        with open("./nlpcc2016ner/" + "train" + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join(seq_result[0:206128]))
            f.close()
        with open("./nlpcc2016ner/" + "dev" + ".txt", "w", encoding='utf-8') as f:
            f.write("\n".join(seq_result[206128:]))
            f.close()
    else:
        raise "error:the data_type should be train or test"
    # q_t_a_df_train.csv和q_t_a_df_test.csv保存的是query，triple和answer的信息
    # q_t_a_df_train.csv保存的是训练集的，q_t_a_df_test.csv是测试集的
