# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/1 16:47

"""
构造属性关联数据集，分类问题，训练 BERT 分类模型
构造数据集的整体属性集合，提取+去重，获得 4373 个属性 RelationList
一个 sample 由 问题+属性+label 构成，原始数据中的属性值为 1
从 RelationList 中随机抽取5个属性作为 Negative Samples
"""

import random
import pandas as pd
from data.load_dbdata import search_data

attribute_classify_sample = []
for data_type in ["train", "test"]:
    if data_type == "train":
        file = "../nlpcc2016/training_clean_triple.csv"
    elif data_type == "test":
        file = "../nlpcc2016/testing_clean_triple.csv"
    else:
        raise "error:the data_type should be train or test"

    # count the number of attribute
    testing_df = pd.read_csv(file, encoding='utf-8')
    attribute_list = testing_df['attribute'].tolist()
    # print(len(attribute_list))  # 14450/9671
    # print(len(set(attribute_list)))  # 4533/4547

    # construct sample
    for row in testing_df.index:
        question, pos_att = testing_df.loc[row][['question', 'attribute']]
        question = question.strip()
        pos_att = pos_att.strip()
        entity = str(testing_df.loc[row]['entity']).strip()
        try:
            sql_entity = "select attribute from nlpccqa where entity= '" + entity + "' order by length(entity)"
            entity_attribute_list = []
            for item in list(search_data(sql_entity)):
                if str(item[0]) != pos_att:
                    entity_attribute_list.append(str(item[0]))
            # print(entity_attribute_list)
        except:
            entity_attribute_list = []
            print(entity)
        # random.shuffle(attribute_list)    the complex is big
        # neg_att_list = attribute_list[0:5]
        if len(entity_attribute_list) < 3:
            neg_att_list = entity_attribute_list
            neg_att_list.extend(random.sample(attribute_list, 3 - len(entity_attribute_list)))
        else:
            neg_att_list = random.sample(entity_attribute_list, 3)
        attribute_classify_sample.append([question, pos_att, '1'])
        neg_att_sample = [[question, neg_att, '0'] for neg_att in neg_att_list if neg_att != pos_att]
        attribute_classify_sample.extend(neg_att_sample)
    seq_result = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(attribute_classify_sample)]
    print(len(seq_result))
    # question_attribute.extend(seq_result)
    # print(len(question_attribute))
    # print(seq_result[0])  # 输出一个样本
    if data_type == 'test':  # 58016
        with open("./Ner_filtered_base3/test.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(seq_result))
            f.close()

    elif data_type == "train":  # 86684
        training_seq_result = seq_result[0:len(seq_result) // 6 * 5]
        with open("./Ner_filtered_base3/train.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(training_seq_result))
            f.close()

        val_seq_result = seq_result[len(seq_result) // 6 * 5:]
        with open("./Ner_filtered_base3/dev.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(val_seq_result))
            f.close()
    else:
        raise "error:the data_type should be train or test"
