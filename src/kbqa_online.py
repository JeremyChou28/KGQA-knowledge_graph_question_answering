# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/2 10:21

import os
import time
import numpy as np
import pandas as pd
import codecs
import pickle
import difflib
import tensorflow as tf
from datetime import time, timedelta, datetime

from data.load_dbdata import search_data
from global_config import Logger
from run_similarity import BertSim
from bert_bilstm_crf_ner_predict import convert_single_example, convert_id_to_label, strage_combined_link_org_loc
from bert_bilstm_crf import FLAGS
from bert_bilstm_crf import create_model, InputFeatures, InputExample  # 基于BERT+BiLSTM+CRF的NER识别
from bert import tokenization  # 导入分词
from bert import modeling  # 导入bert模型

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


def online_KBQA():
    ''' KBQA在线测试 '''

    def convert(line):
        feature = convert_single_example(0, line, label_list, FLAGS.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, FLAGS.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, FLAGS.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, FLAGS.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, FLAGS.max_seq_length))
        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        # print(id2label)
        bs = BertSim()
        bs.set_mode(tf.estimator.ModeKeys.PREDICT)
        while True:
            print('input the question:（exit/退出）')
            sentence = input()
            if sentence == "exit" or sentence == "退出":
                break
            question = sentence
            if len(sentence) < 2:
                print("输入问题太短，无法检索，请重新输入！")
                continue
            sentence = tokenizer.tokenize(sentence)
            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         segment_ids_p: segment_ids,
                         label_ids_p: label_ids}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            # print(pred_label_result)
            result = strage_combined_link_org_loc(sentence, pred_label_result[0], True)
            print('识别的实体有：{}'.format(''.join(result)))

            # 未识别到实体，对原query进行非语义匹配
            if result == []:
                print("抱歉没有检索到该问题的实体！")
                sql_question = "select id,answer,question from nlpccqa"
                result_question = list(search_data(sql_question))
                result_df = pd.DataFrame(result_question, columns=['id', 'answer', 'question'])
                attribute_candicate_sim = [(k, edit_distance(question, k)) for k in
                                           result_df['question'].tolist()]
                attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1])
                sim_question_index = result_df[result_df.question == attribute_candicate_sort[0][0]].index.tolist()[0]
                answer = result_df.answer.loc[sim_question_index]
                print("您可能要查询的问题是：", attribute_candicate_sort[0][0], "\t该问题的答案是：", answer)
                continue

            # 识别到实体
            total = 0
            recall = 0

            try:
                total += 1
                ner = result[0]
                ner = ner.replace("#", "").replace("[UNK]", "%").replace("\n", "")
                # case: entity Fuzzy Match
                # 找出所有包含这些实体的三元组
                sql_entity = "select entity,attribute,answer,question from nlpccQA where entity like '%" + ner + "%' order by length(entity) asc limit 20"
                sql_answer = "select entity,attribute,answer,question from nlpccQA where answer like '%" + ner + "%' order by length(entity) asc limit 20"
                # sql_relation = "select entity,attribute,answer,question from nlpccQA where attribute like '%" + ner + "%' order by length(entity) asc limit 20"
                # sql查出来的是tuple，要转换成list才不会报错
                result_e = list(search_data(sql_entity))
                result_a = list(search_data(sql_answer))
                # result_r = list(search_data(sql_relation))

                # 数据库检索，得到推荐的答案结果
                # 在数据库中检索到相应实体
                if len(result_e) > 0:
                    recall += 1
                    # 非语义匹配
                    flag_fuzzy = True
                    # 非语义匹配中属性和问题的相似度字典，key为属性，value为相似度值
                    string_similar_value_dict = {}
                    for l in result_e:
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
                        for l in result_e:
                            if l[1] == key:
                                print("非语义匹配成功，问题是：", question, "\t答案为：", l[2])
                    # 属性非语义匹配失败，利用bert进行文本分类计算相似度
                    else:
                        # 语义匹配
                        result_df = pd.DataFrame(result_e,
                                                 columns=['entity', 'attribute', 'answer', 'question'])

                        attribute_candicate_sim = [(k, bs.predict(question, k)[0][1]) for k in
                                                   result_df['attribute'].tolist()]
                        attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1],
                                                          reverse=True)
                        sim_question_index = \
                            result_df[result_df.attribute == attribute_candicate_sort[0][0]].index.tolist()[0]
                        answer = result_df.answer.loc[sim_question_index]
                        print("语义匹配成功，匹配问题是：", result_df.question.loc[sim_question_index], "\t答案为：", answer)

                elif len(result_a) > 0:
                    recall += 1
                    # 非语义匹配
                    flag_fuzzy = True
                    # 非语义匹配中属性和问题的相似度字典，key为属性，value为相似度值
                    string_similar_value_dict = {}
                    for l in result_a:
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
                        for l in result_a:
                            if l[1] == key:
                                print("非语义匹配成功，问题是：", question, "\t答案为：", l[0])
                    # 属性非语义匹配失败，利用bert进行文本分类计算相似度
                    else:
                        # 语义匹配
                        result_df = pd.DataFrame(result_a,
                                                 columns=['entity', 'attribute', 'answer', 'question'])

                        attribute_candicate_sim = [(k, bs.predict(question, k)[0][1]) for k in
                                                   result_df['attribute'].tolist()]
                        attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1],
                                                          reverse=True)
                        sim_question_index = \
                            result_df[result_df.attribute == attribute_candicate_sort[0][0]].index.tolist()[0]
                        answer = result_df.answer.loc[sim_question_index]
                        print("语义匹配成功，匹配问题是：", result_df.question.loc[sim_question_index], "\t答案为：", answer)
                # 在数据库中未检索到相应实体
                else:
                    print("抱歉没有检索到\t" + question + "\t的结果！")
                    sql_question = "select id,answer,question from nlpccqa"
                    result_question = list(search_data(sql_question))
                    result_df = pd.DataFrame(result_question, columns=['id', 'answer', 'question'])
                    attribute_candicate_sim = [(k, edit_distance(question, k)) for k in
                                               result_df['question'].tolist()]
                    attribute_candicate_sort = sorted(attribute_candicate_sim, key=lambda candicate: candicate[1])
                    sim_question_index = result_df[result_df.question == attribute_candicate_sort[0][0]].index.tolist()[
                        0]
                    answer = result_df.answer.loc[sim_question_index]
                    print("您可能要查询的问题是：", result_df.question.loc[sim_question_index], "\t答案为：", answer)

            except Exception as e:
                loginfo.logger.info("the question id % d occur error %s" % (total, repr(e)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    loginfo = Logger("../log/kbqa_online.log", "info")
    file = "../nerpredict/nlpcc2016nerpredict/q_t_a_testing_predict.txt"

    # NER训练后的checkpoint保存地址
    print(FLAGS.output_dir)
    print('checkpoint path:{}'.format(os.path.join(FLAGS.output_dir, "checkpoint")))
    if not os.path.exists(os.path.join(FLAGS.output_dir, "checkpoint")):
        raise Exception("failed to get checkpoint. going to return ")

    # label2id.pkl和label_list.pkl是NER训练保存的文件
    # 加载label->id的词典
    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'rb') as rf:
        label_list = pickle.load(rf)
    num_labels = len(label_list) + 1

    graph = tf.get_default_graph()
    with graph.as_default():
        print("going to restore checkpoint")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="input_mask")
        label_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="label_ids")
        segment_ids_p = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name="segment_ids")

        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
            label_ids_p, num_labels, use_one_hot_embeddings)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.output_dir))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    online_KBQA()
