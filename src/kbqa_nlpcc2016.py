# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/2 10:21

import sys
import os
import time
import numpy as np
import pandas as pd
import codecs
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
    recall = 0
    correct = 0

    semantic_match = []
    nonsemantic_match = []
    not_match = []
    semantic_error = []
    semantic_true = []
    semantic_examples = []
    nonsemantic_error = []
    nonsemantic_true = []
    notmatch_error = []
    notmatch_true = []
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

            # 数据库检索，得到推荐的答案结果
            # 在数据库中检索到相应实体
            if len(result_e0_a1) > 0:
                recall += 1
                # 非语义匹配
                flag_fuzzy = True
                # 非语义匹配中属性和问题的相似度字典，key为属性，value为相似度值
                string_similar_value_dict = {}
                for l in result_e0_a1:
                    # 如果实体匹配后的三元组中的属性也在question中出现，则非语义匹配成功
                    if l[1] in question or l[1].lower() in question or l[1].upper() in question:
                        flag_fuzzy = False
                        # 计算question与属性的相似度
                        string_similar_value_dict[l[1]] = edit_distance(question, l[1])

                # 属性的非语义匹配成功，继续下一次
                if not flag_fuzzy:
                    def get_keys(d, value):
                        return [k for k, v in d.items() if v == value]

                    best_similar = min(string_similar_value_dict.values())
                    key = get_keys(string_similar_value_dict, best_similar)[0]
                    for l in result_e0_a1:
                        if l[1] == key:
                            if l[2] == answer:
                                correct += 1
                                accuracy = correct * 100.0 / total
                                averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                                        recall * 100.0 / total + accuracy)
                                nonsemantic_true.append((question, entity, attribute, answer))
                                loginfo.logger.info(
                                    "非语义匹配预测成功，问题是" + question + "预测答案为：" + l[2] +
                                    "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1: {}%".format(
                                        total,
                                        recall,
                                        correct,
                                        accuracy,
                                        averagedF1))
                            else:
                                nonsemantic_error.append((question, entity, attribute, answer, l[2]))
                                accuracy = correct * 100.0 / total
                                averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                                        recall * 100.0 / total + accuracy)
                                loginfo.logger.info(
                                    "非语义匹配预测失败，问题是" + question + "预测答案为：" + l[2] +
                                    "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1：{}%".format(
                                        total,
                                        recall,
                                        correct,
                                        accuracy,
                                        averagedF1))
                            break
                    continue
                    # print("非语义匹配成功，", question, "答案为：", l[2])
                # 属性非语义匹配失败，利用bert进行文本分类计算相似度
                else:
                    # 语义匹配
                    result_df = pd.DataFrame(result_e0_a1,
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
                        averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                                recall * 100.0 / total + accuracy)
                        semantic_true.append((question, entity, attribute, answer))
                        loginfo.logger.info(
                            "语义匹配预测成功，问题是：" + question + "预测答案为：" + predict_answer +
                            "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1：{}%".format(
                                total,
                                recall,
                                correct,
                                accuracy,
                                averagedF1))
                    else:
                        accuracy = correct * 100.0 / total
                        averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                                recall * 100.0 / total + accuracy)
                        semantic_error.append((question, entity, attribute, answer, predict_answer))
                        loginfo.logger.info(
                            "语义匹配预测失败，问题是：" + question + "预测答案为：" + predict_answer +
                            "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1：{}%".format(
                                total,
                                recall,
                                correct,
                                accuracy,
                                averagedF1))
                    continue
                    # print("语义匹配成功，", result_df.question.loc[sim_question_index], "答案为：", answer)
            # 在数据库中未检索到相应实体
            else:
                sql_question = "select id,answer,entity from nlpccqa"
                result_question = list(search_data(sql_question))
                result_df = pd.DataFrame(result_question, columns=['id', 'answer', 'entity'])
                entity_candicate_sim = [(k, edit_distance(ner, k)) for k in
                                        result_df['entity'].tolist()]
                entity_candicate_sort = sorted(entity_candicate_sim, key=lambda candicate: candicate[1])
                sim_entity_index = result_df[result_df.entity == entity_candicate_sort[0][0]].index.tolist()[
                    0]
                predict_answer = result_df.answer.loc[sim_entity_index]
                not_match.append((question, result_df.question.loc[sim_entity_index], predict_answer, answer))
                if predict_answer == answer:
                    correct += 1
                    accuracy = correct * 100.0 / total
                    averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                            recall * 100.0 / total + accuracy)
                    notmatch_true.append((question, entity, attribute, answer))
                    loginfo.logger.info(
                        "实体链接成功，问题是：" + question + "预测答案为：" + predict_answer +
                        "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1：{}%".format(
                            total,
                            recall,
                            correct,
                            accuracy,
                            averagedF1))
                else:
                    accuracy = correct * 100.0 / total
                    averagedF1 = (2 * (recall * 100.0 / total * accuracy)) / (
                            recall * 100.0 / total + accuracy)
                    notmatch_error.append((question, entity, attribute, answer, predict_answer))

                    loginfo.logger.info(
                        "抱歉没有检索到" + question + "该问题的结果！您可能要查询的问题是：" + result_df.question.loc[
                            sim_entity_index] + "\t预测答案是：" +
                        predict_answer +
                        "\ntotal: {}, recall: {}, correct:{}, accuracy: {}%, averagedF1：{}%".format(
                            total,
                            recall,
                            correct,
                            accuracy,
                            averagedF1))
                # print("您可能要查询的问题是：", result_df.question.loc[sim_question_index], "\t该问题的答案是：", answer)

        except Exception as e:
            loginfo.logger.info("the question id % d occur error %s" % (total, repr(e)))

    df1 = pd.DataFrame(semantic_match, columns=["question", "predict_answer", "accurate_answer"])
    df1.to_csv("../qaresults/originnlpcc2016/" + results_path + "/semantic_match.csv", encoding='utf-8', index=False)

    df2 = pd.DataFrame(nonsemantic_match, columns=["question", "predict_answer", "accurate_answer"])
    df2.to_csv("../qaresults/originnlpcc2016/" + results_path + "/nonsemantic_match.csv", encoding='utf-8', index=False)

    df3 = pd.DataFrame(not_match, columns=["raw_question", "match_question", "predict_answer", "accurate_answer"])
    df3.to_csv("../qaresults/originnlpcc2016/" + results_path + "/not_match.csv", encoding='utf-8', index=False)

    df4 = pd.DataFrame(nonsemantic_true, columns=["question", "entity", "attribute", "answer"])
    df4.to_csv("../qaresults/originnlpcc2016/" + results_path + "/nonsemantic_true.csv", encoding='utf-8', index=False)

    df5 = pd.DataFrame(nonsemantic_error, columns=["question", "entity", "attribute", "answer", "predict_answer"])
    df5.to_csv("../qaresults/originnlpcc2016/" + results_path + "/nonsemantic_error.csv", encoding='utf-8', index=False)

    df6 = pd.DataFrame(semantic_true, columns=["question", "entity", "attribute", "answer"])
    df6.to_csv("../qaresults/originnlpcc2016/" + results_path + "/semantic_true.csv", encoding='utf-8', index=False)

    df7 = pd.DataFrame(semantic_error, columns=["question", "entity", "attribute", "answer", "predict_answer"])
    df7.to_csv("../qaresults/originnlpcc2016/" + results_path + "/semantic_error.csv", encoding='utf-8', index=False)

    df6 = pd.DataFrame(notmatch_true, columns=["question", "entity", "attribute", "answer"])
    df6.to_csv("../qaresults/originnlpcc2016/" + results_path + "/notmatch_true.csv", encoding='utf-8', index=False)

    df7 = pd.DataFrame(notmatch_error, columns=["question", "entity", "attribute", "answer", "predict_answer"])
    df7.to_csv("../qaresults/originnlpcc2016/" + results_path + "/notmatch_error.csv", encoding='utf-8', index=False)

    with open("../qaresults/originnlpcc2016/" + results_path + "/semantic_examples.pkl", 'wb') as fo:
        pickle.dump(semantic_examples, fo)
        fo.close()


def single_test(line):
    """
    对单个问答句子对做测试
    """
    Fi_list = []
    try:
        question, entity, attribute, answer, ner = line.split("\t")
        answer = ''.join(answer.split())
        ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
        # print([ner])
        # case: entity Fuzzy Match
        # 找出所有包含这些实体的三元组
        sql_e0_a1 = "select entity,attribute,answer,question from nlpccqa where entity like '%" + ner + "%' order by length(entity)"
        # sql查出来的是tuple，要转换成list才不会报错，每一个记录为[entity,attribute,answer,question]
        result_e0_a1 = list(search_data(sql_e0_a1))
        # print(len(result_e0_a1))

        sql_ci = "select id,entity from nlpccqa where entity like '%" + ner + "%'"
        result_ci = list(search_data(sql_ci))
        sql_ai = "select id,answer from nlpccqa where answer=" + answer
        result_ai = list(search_data(sql_ai))
        if len(result_ci) == 0:
            Fi_list.append(0)
        else:
            ci_set = []
            ai_set = []
            for item in result_ci:
                ci_set.append(item[0])
            for item in result_ai:
                ai_set.append(item[0])
            common = [x for x in ci_set if x in ai_set]
            fi = 2 * (len(common) ** 2) / (
                    len(ci_set) * len(ai_set) * (len(common) / len(ci_set) + len(common) / len(ai_set)))
            print(fi)
            Fi_list.append(fi)
        # 数据库检索，得到推荐的答案结果
        # 在数据库中检索到相应实体
        if len(result_e0_a1) > 0:
            # 非语义匹配
            flag_fuzzy = True
            # 非语义匹配中属性和问题的相似度字典，key为属性，value为相似度值
            string_similar_value_dict = {}
            for l in result_e0_a1:
                # 如果实体匹配后的三元组中的属性也在question中出现，则非语义匹配成功
                if l[1] in question or l[1].lower() in question or l[1].upper() in question:
                    flag_fuzzy = False
                    # 计算question与属性的相似度
                    string_similar_value_dict[l[1]] = edit_distance(question, l[1])

            # 属性的非语义匹配成功，继续下一次
            if not flag_fuzzy:
                def get_keys(d, value):
                    return [k for k, v in d.items() if v == value]

                best_similar = min(string_similar_value_dict.values())
                key = get_keys(string_similar_value_dict, best_similar)[0]
                for l in result_e0_a1:
                    if l[1] == key:
                        predict_answer = ''.join(l[2].split())
                        if predict_answer == answer:
                            loginfo.logger.info('非语义答案预测正确\t' + line)
                        else:
                            loginfo.logger.info('非语义答案预测错误\t' + line)
                        break
                # print("非语义匹配成功，", question, "答案为：", l[2])
            # 属性非语义匹配失败，利用bert进行文本分类计算相似度
            else:
                # 语义匹配
                result_df = pd.DataFrame(result_e0_a1,
                                         columns=['entity', 'attribute', 'answer', 'question'])
                attribute_candicate_sim = []
                for index in range(len(result_df)):
                    h = result_df.loc[index]['entity']
                    k = result_df.loc[index]['attribute']
                    t = result_df.loc[index]['answer']
                    attribute_candicate_sim.append((h, k, t, bs.predict(question, k)[0][1]))
                attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[3],
                                                  reverse=True)
                sim_question_index = \
                    result_df[result_df.attribute == attribute_candicate_sort[0][1]].index.tolist()[0]
                predict_answer = result_df.answer.loc[sim_question_index]
                if predict_answer == answer:
                    loginfo.logger.info('语义答案预测正确\t' + line)
                else:
                    loginfo.logger.info(
                        '语义答案预测错误' + '\t' + question + '\t' + answer + '\t' + result_df.question.loc[
                            sim_question_index] + '\t' + predict_answer)
                loginfo.logger.info(str(attribute_candicate_sim))
                # error_semantic.append(
                #     [question + '\t' + entity + '\t' + attribute + '\t' + answer, result_df.question.loc[
                #         sim_question_index] + '\t' + ner + '\t' + result_df.attribute.loc[
                #          sim_question_index] + '\t' + predict_answer])
                # print("语义匹配成功，", result_df.question.loc[sim_question_index], "答案为：", answer)
        # 在数据库中未检索到相应实体
        else:
            # print("抱歉没有检索到该问题的结果！")
            sql_question = "select id,answer,question from nlpccqa"
            result_question = list(search_data(sql_question))
            result_df = pd.DataFrame(result_question, columns=['id', 'answer', 'question'])
            attribute_candicate_sim = [(k, edit_distance(question, k)) for k in
                                       result_df['question'].tolist()]
            attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1])
            sim_question_index = result_df[result_df.question == attribute_candicate_sort[0][0]].index.tolist()[
                0]
            predict_answer = result_df.answer.loc[sim_question_index]
            predict_answer = ''.join(predict_answer.split())
            if predict_answer == answer:
                loginfo.logger.info('识别实体未在知识库中预测正确\t' + line)
            else:
                loginfo.logger.info('识别实体未在知识库中预测错误\t' + line)
            # print("您可能要查询的问题是：", result_df.question.loc[sim_question_index], "\t该问题的答案是：", answer)
    except Exception as e:
        loginfo.logger.info("the question occur error %s" % (repr(e)))


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start_time = time.time()

    loginfo = Logger("../log/originnlpcc2016/base3_kbqa.log", "info")
    line_list = read_file('../nerpredict/originnlpcc2016nerpredict/q_t_a_testing_predict.txt')
    results_path = 'base3'
    lines = []
    for i in range(len(line_list)):
        question, entity, attribute, answer, ner = line_list[i]
        if ner == '\n':
            continue
        line = str(question) + '\t' + str(entity) + '\t' + str(attribute) + '\t' + str(answer) + '\t' + str(ner)
        lines.append(line)
        # single_test(line)
    # print(len(lines))
    # print(lines[6332])
    outline_KBQA(lines, results_path)
    loginfo.logger.info("the program spend " + str(time.time() - start_time))

    """
    # 单句测试
    loginfo = Logger("../log/kbqa_outline2.log", "info")
    line = "你知道计算机应用基础这本书的作者是谁吗？	计算机应用基础	作者	秦婉，王蓉	计算机应用基础"
    question, entity, attribute, answer, ner = line.split('\t')
    print([question, entity, attribute, answer, ner])
    single_test(line)
    loginfo.logger.info("the program spend " + str(time.time() - start_time))
    """

    """
    line_list = read_file('./Data/NER_Data/ner_predict_true.txt')
    error_semantic = []
    for i in range(len(line_list)):
        question, entity, attribute, answer, ner = line_list[i]
        line = str(question) + '\t' + str(entity) + '\t' + str(attribute) + '\t' + str(answer) + '\t' + str(ner)
        single_test(line)
    with open('./output_bert_bilstm_crf/error_semantic.txt', 'w+', encoding='utf-8') as f:
        for item in error_semantic:
            f.write(item[0] + '\n' + item[1] + '\n')
            f.write('-' * 50 + '\n')
    loginfo.logger.info("the program spend " + str(time.time() - start_time))
    """
