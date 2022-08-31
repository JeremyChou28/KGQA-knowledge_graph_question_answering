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

pos_examples = []
neg_examples = []
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
    print(data_type, "中属性个数:", len(attribute_list))  # train中属性个数为14609，test中为9870个
    print(data_type, "中去重后属性个数:", len(set(attribute_list)))  # train中去重后得到4533个属性，test中去重后得到4551个属性
    print("输出属性列表的前5个item:", attribute_list[0:5])
    # 查看前5行信息
    # print(testing_df.head())

    # construct sample
    for row in testing_df.index:
        question, pos_att = testing_df.loc[row][['question', 'attribute']]
        question = question.strip()  # 问题
        pos_att = pos_att.strip()  # 问题属性

        # random.shuffle(attribute_list)    the complex is big
        # neg_att_list = attribute_list[0:5]

        # 从属性集中attribute_list中随机抽取5个属性作为negative samples
        random.seed(0)
        neg_att_list = random.sample(attribute_list, 5)
        # 构建samples，一个sample由 问题+属性+标签 组成，原始sample标签为1，negative sample标签为0
        pos_examples.append([question, pos_att, '1'])
        neg_att_sample = [[question, neg_att, '0'] for neg_att in neg_att_list if neg_att != pos_att]
        neg_examples.extend(neg_att_sample)

random.shuffle(pos_examples)
random.shuffle(neg_examples)
pos_length = len(pos_examples)
neg_length = len(neg_examples)

train_samples = pos_examples[:pos_length // 2] + (neg_examples[:neg_length // 2])
dev_samples = pos_examples[pos_length // 2:pos_length // 10 + pos_length // 2] + (
    neg_examples[neg_length // 2:neg_length // 10 + neg_length // 2])
test_samples = pos_examples[pos_length // 10 + pos_length // 2:] + (
    neg_examples[neg_length // 10 + neg_length // 2:])
random.shuffle(train_samples)
random.shuffle(dev_samples)
random.shuffle(test_samples)
# seq_result = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(attribute_classify_sample)]
train_lines = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(train_samples)]
dev_lines = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(dev_samples)]
test_lines = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(test_samples)]
print("数据集样本总量：", len(pos_examples + neg_examples))
print("训练集样本数量：", len(train_samples))
print("验证集样本数量：", len(dev_samples))
print("测试集样本数量：", len(test_samples))

with open("./Base5_shuffle/train.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(train_lines))
    f.close()
with open("./Base5_shuffle/dev.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(dev_lines))
    f.close()
with open("./Base5_shuffle/test.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(test_lines))
    f.close()

"""
if data_type == 'test':
    with open("./Sim_Data/" + data_type + ".csv", "w", encoding='utf-8') as f:
        f.write("\n".join(seq_result))
        f.close()
elif data_type == "train":
    val_seq_result = seq_result[0:12000]
    with open("./Sim_Data/" + "dev" + ".csv", "w", encoding='utf-8') as f:
        f.write("\n".join(val_seq_result))
        f.close()
    training_seq_result = seq_result[12000:]
    with open("./Sim_Data/" + data_type + ".csv", "w", encoding='utf-8') as f:
        f.write("\n".join(training_seq_result))
        f.close()
else:
    raise "error:the data_type should be train or test"
# 构建的数据集分为train.csv,dev.csv,test.csv，为训练集、开发集、测试集
# 数据集中的每个样本由序号，question，attribute和标签组成，标签为0代表负样本，为1代表正样本
# 最后训练集中有75644个样本，开发集中有12000个样本，测试集中有59209个样本
"""
