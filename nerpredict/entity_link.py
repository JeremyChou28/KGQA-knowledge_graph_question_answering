# @description:分析实体识别预测结果ner_predict_true中有多少是错误的
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/8/5 11:37


import sys
import os
import numpy as np
import difflib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from data.load_dbdata import search_data


def find_error(file):
    with open('ner_entity_error.txt', 'w+', encoding='utf-8') as f:
        with open(file, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            id_list = []
            id = 1
            line_list = []
            for line in lines:
                question, entity, attribute, answer, ner = line.split('\t')
                if entity != ner.strip():
                    id_list.append(id)
                    line_list.append(line)
                    f.write(line)
                id += 1
            f1.close()
        f.close()
        print(id_list)
        print(len(id_list))


def string_similar(s1, s2):
    '''计算两个字符串之间的相似度'''
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def edit_distance(word1, word2):
    '''计算两个字符串之间的编辑距离'''
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


def entity_link(file):
    sql_entity = "select id,entity from webqa"
    result_entity = list(search_data(sql_entity))
    result_df = pd.DataFrame(result_entity, columns=['id', 'entity'])
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        originerror = []
        linked_true = []
        linked_error = []
        for line in lines:
            question, entity, attribute, answer, ner = line.split('\t')
            if entity != ner.strip():
                originerror.append(line)
                # entity_candicate_sim = [(k, 0.5 * string_similar(ner, str(k)) + 0.5 * string_similar(question, str(k)))
                #                         for k in
                #                         result_df['entity'].tolist()]
                entity_candicate_sim = [(k, string_similar(ner, str(k))) for k in
                                        result_df['entity'].tolist()]
                entity_candicate_sort = sorted(entity_candicate_sim, key=lambda candicate: candicate[1])
                sim_entity_index = result_df[result_df.entity == entity_candicate_sort[0][0]].index.tolist()[
                    0]
                linked_ner = result_df.entity.loc[sim_entity_index]
                if linked_ner == entity:
                    linked_true.append(line)
                    # print(line.strip(), linked_ner)
                else:
                    linked_error.append(line)
                    print(line.strip(), linked_ner)
        f.close()
    print(len(originerror))
    print(len(linked_true))
    print(len(linked_error))


def correct_dataset(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
        f1.close()
    with open(file2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
        f2.close()
    lines1_list = []
    lines2_list = []
    for line in lines1:
        question, entity, attribute, answer, ner = line.split('\t')
        lines1_list.append([question, entity, attribute, answer, ner])
    for line in lines2:
        question, entity, attribute, answer, ner = line.split('\t')
        lines2_list.append([question, entity, attribute, answer, ner])
    df1 = pd.DataFrame(lines1_list, columns=['question', 'entity', 'attribute', 'answer', 'ner'])
    df2 = pd.DataFrame(lines2_list, columns=['question', 'entity', 'attribute', 'answer', 'ner'])
    questions1 = df1['question'].tolist()
    questions2 = df2['question'].tolist()
    correct_list = []
    for item in questions1:
        if item not in questions2:
            correct_list.append(item)
            print(item)
    print(len(correct_list))


if __name__ == "__main__":
    file = './webqanerpredict/bert_bilstm_crf/ner_predict_error.txt'
    entity_link(file)

    # file1 = 'q_t_a_testing_predict.txt'
    # file2 = '../nlpcc2016nerpredict/q_t_a_testing_predict.txt'
    # correct_dataset(file1, file2)
