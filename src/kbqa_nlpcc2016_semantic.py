# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/2 10:21

import sys
import os
import time
import numpy as np
import pandas as pd
import pickle
import difflib
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from data.load_dbdata import search_data
from global_config import Logger
from run_similarity import BertSim

is_training = False
use_one_hot_embeddings = False
batch_size = 1

# 动态申请显存
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
model = None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

bs = BertSim()
bs.set_mode(tf.estimator.ModeKeys.PREDICT)


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


def estimate_answer(candidate, answer):
    '''
    :param candidate:
    :param answer:
    :return:
    '''
    candidate = candidate.strip().lower()
    answer = answer.strip().lower()
    if candidate == answer:
        return True

    if not answer.isdigit() and candidate.isdigit():
        candidate_temp = "{:.5E}".format(int(candidate))
        if candidate_temp == answer:
            return True
        candidate_temp == "{:.4E}".format(int(candidate))
        if candidate_temp == answer:
            return True
    return False


def outline_KBQA(lines, results_path):
    """
    进行问答测试：
    1、 实体检索:输入问题，ner得出实体集合，在数据库中检索与输入实体相关的所有三元组
    2、 属性映射——bert分类/文本相似度
        + 非语义匹配：如果所得三元组的关系(attribute)属性是 输入问题 字符串的子集，将所得三元组的答案(answer)属性与正确答案匹配，correct +1
        + 语义匹配：利用bert计算输入问题(input question)与所得三元组的关系(attribute)属性的相似度，将最相似的三元组的答案作为答案，并与正确
          的答案进行匹配，correct +1
    3、 答案组合
    :return:
    """
    total = 0
    correct = 0

    semantic_match = []
    semantic_error = []
    semantic_true = []
    semantic_examples = []
    for line in lines:
        try:
            total += 1
            question, entity, attribute, answer, ner = line.split("\t")
            # answer = ''.join(answer.split())
            ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
            # case: entity Fuzzy Match
            # 找出所有包含这些实体的三元组
            sql_e0_a1 = "select entity,attribute,answer,question from nlpccqa where entity like '%" + ner + "%' order by length(entity)"
            # sql查出来的是tuple，要转换成list才不会报错，每一个记录为[entity,attribute,answer,question]
            result_e0_a1 = list(search_data(sql_e0_a1))
            candicate_triples = result_e0_a1
            if len(result_e0_a1) == 0:
                sql_entity = "select id,entity from nlpccqa"
                result_entity = list(search_data(sql_entity))
                result_df = pd.DataFrame(result_entity, columns=['id', 'entity'])
                entity_candicate_sim = [(k, edit_distance(ner, str(k))) for k in
                                        result_df['entity'].tolist()]
                entity_candicate_sort = sorted(entity_candicate_sim, key=lambda candicate: candicate[1])
                sim_entity_index = result_df[result_df.entity == entity_candicate_sort[0][0]].index.tolist()[
                    0]
                linked_ner = result_df.entity.loc[sim_entity_index]
                sql_e0_a1 = "select entity,attribute,answer,question from nlpccqa where entity like '%" + linked_ner + "%' order by length(entity)"
                # sql查出来的是tuple，要转换成list才不会报错，每一个记录为[entity,attribute,answer,question]
                candicate_triples = list(search_data(sql_e0_a1))

            # sql_ci = "select id,entity from nlpccqa where entity like '%" + ner + "%'"
            # result_ci = list(search_data(sql_ci))
            # sql_ai = "select id,answer from nlpccqa where answer='" + answer + "'"
            # result_ai = list(search_data(sql_ai))
            if len(candicate_triples) > 0:
                # ci_set = []
                # ai_set = []
                # for item in result_ci:
                #     ci_set.append(item[0])
                # for item in result_ai:
                #     ai_set.append(item[0])
                # common = [x for x in ci_set if x in ai_set]
                # fi = 200 * (len(common) ** 2) / (
                #         len(ci_set) * len(ai_set) * (len(common) / len(ci_set) + len(common) / len(ai_set)))
                # F1_list.append(fi)

                # 语义匹配
                result_df = pd.DataFrame(candicate_triples,
                                         columns=['entity', 'attribute', 'answer', 'question'])
                attribute_candicate_sim = []
                for index in range(len(result_df)):
                    h = result_df.loc[index]['entity']
                    k = result_df.loc[index]['attribute']
                    t = result_df.loc[index]['answer']
                    attribute_candicate_sim.append((h, k, t, bs.predict(question, k)[0][1]))
                semantic_examples.append(attribute_candicate_sim)
                attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[3],
                                                  reverse=True)
                sim_question_index = \
                    result_df[result_df.attribute == attribute_candicate_sort[0][1]].index.tolist()[0]
                predict_answer = result_df.answer.loc[sim_question_index]
                semantic_match.append((question, predict_answer, answer))
                if predict_answer == answer:
                    correct += 1
                    accuracy = correct * 100.0 / total
                    semantic_true.append((question, entity, attribute, answer))
                    loginfo.logger.info(
                        "语义匹配预测成功，问题是：" + question + "预测答案为：" + predict_answer +
                        "\ntotal: {}, correct:{}, accuracy: {}%".format(
                            total,
                            correct,
                            accuracy))
                else:
                    accuracy = correct * 100.0 / total
                    semantic_error.append((question, entity, attribute, answer, predict_answer))
                    loginfo.logger.info(
                        "语义匹配预测失败，问题是：" + question + "预测答案为：" + predict_answer +
                        "\ntotal: {}, correct:{}, accuracy: {}%".format(
                            total,
                            correct,
                            accuracy))
        except Exception as e:
            loginfo.logger.info("the question id % d occur error %s" % (total, repr(e)))

    df1 = pd.DataFrame(semantic_match, columns=["question", "predict_answer", "accurate_answer"])
    df1.to_csv("../qaresults/nlpcc2016/" + results_path + "/semantic_match.csv", encoding='utf-8', index=False)

    df2 = pd.DataFrame(semantic_true, columns=["question", "entity", "attribute", "answer"])
    df2.to_csv("../qaresults/nlpcc2016/" + results_path + "/semantic_true.csv", encoding='utf-8', index=False)

    df3 = pd.DataFrame(semantic_error, columns=["question", "entity", "attribute", "answer", "predict_answer"])
    df3.to_csv("../qaresults/nlpcc2016/" + results_path + "/semantic_error.csv", encoding='utf-8', index=False)

    with open("../qaresults/nlpcc2016/" + results_path + "/semantic_examples.pkl", 'wb') as fo:
        pickle.dump(semantic_examples, fo)
        fo.close()


def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_list = []
        for i in range(len(lines)):
            line = lines[i]
            line_list.append(line.split('\t'))
        f.close()
    return line_list


def read_test(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_list = []
        for line in lines:
            line_list.append(line.split('\t'))
        f.close()
    return line_list


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    start_time = time.time()

    loginfo = Logger("../log/nlpcc2016/Ner_filtered_base3_kbqa.log", "info")
    line_list = read_file('../nerpredict/nlpcc2016nerpredict/q_t_a_testing_predict.txt')
    results_path = 'Ner_filtered_base3'
    lines = []
    for i in range(len(line_list)):
        question, entity, attribute, answer, ner = line_list[i]
        if ner == '\n':
            continue
        line = str(question) + '\t' + str(entity) + '\t' + str(attribute) + '\t' + str(answer) + '\t' + str(ner)
        lines.append(line)
    outline_KBQA(lines, results_path)
    loginfo.logger.info("the program spend " + str(time.time() - start_time))
