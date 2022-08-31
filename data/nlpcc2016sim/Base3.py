# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/3/30 22:53

"""
构造属性关联数据集，分类问题，训练BERT分类模型
构造数据集的整体属性集合，提取+去重，获得 4373 个属性 RelationList
一个sample由 问题+属性+label 构成，原始数据中的属性值为 1
从 RelationList 中随机抽取5个属性作为 Negative Samples
"""

import random
import pandas as pd

data_type = "train"
if data_type == "train":
    file = "../nlpcc2016/training_clean_triple.csv"
elif data_type == "test":
    file = "../nlpcc2016/testing_clean_triple.csv"
else:
    raise "error:the data_type should be train or test"

attribute_classify_sample = []

# count the number of attribute
testing_df = pd.read_csv(file, encoding='utf-8')
attribute_list = testing_df['attribute'].tolist()
print(data_type, "中属性个数:", len(attribute_list))  # train中属性个数为14609，test中为9870个
print(data_type, "中去重后属性个数:", len(set(attribute_list)))  # train中去重后得到4533个属性，test中去重后得到4550个属性
print("输出属性列表的前5个item:", attribute_list[0:5])

# construct sample
for row in testing_df.index:
    question, pos_att = testing_df.loc[row][['question', 'attribute']]
    question = question.strip()  # 问题
    pos_att = pos_att.strip()  # 问题属性

    # random.shuffle(attribute_list)    the complex is big
    # neg_att_list = attribute_list[0:5]

    # 从属性集中attribute_list中随机抽取5个属性作为negative samples
    neg_att_list = random.sample(attribute_list, 3)
    # 构建samples，一个sample由 问题+属性+标签 组成，原始sample标签为1，negative sample标签为0
    attribute_classify_sample.append([question, pos_att, '1'])
    neg_att_sample = [[question, neg_att, '0'] for neg_att in neg_att_list if neg_att != pos_att]
    attribute_classify_sample.extend(neg_att_sample)

seq_result = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(attribute_classify_sample)]
print(data_type, "中构建的样本数为:", len(seq_result))  # 从training-data中构建了57790个样本，从testing-data中构建了38676个样本

if data_type == 'test':
    with open("./Base3/" + data_type + ".txt", "w", encoding='utf-8') as f:
        f.write("\n".join(seq_result))
        f.close()
elif data_type == "train":
    val_seq_result = seq_result[0:9647]
    with open("./Base3/" + "dev" + ".txt", "w", encoding='utf-8') as f:
        f.write("\n".join(val_seq_result))
        f.close()
    training_seq_result = seq_result[9647:]
    with open("./Base3/" + data_type + ".txt", "w", encoding='utf-8') as f:
        f.write("\n".join(training_seq_result))
        f.close()
else:
    raise "error:the data_type should be train or test"
# 构建的数据集分为train.csv,dev.csv,test.csv，为训练集、开发集、测试集
# 数据集中的每个样本由序号，question，attribute和标签组成，标签为0代表负样本，为1代表正样本
# 最后训练集中有75644个样本，开发集中有12000个样本，测试集中有59209个样本
