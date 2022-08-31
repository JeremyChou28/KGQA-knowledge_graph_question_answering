# @description:
# @author:Jianping Zhou
# @company:Shandong University
# @Time:2022/4/1 16:47

import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.pardir

# model_dir = os.path.join(file_path, 'ModelParams/chinese_L-12_H-768_A-12/')
model_dir = '/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12'
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(file_path, 'checkpoints/webqa/sim/Base5')
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(file_path, 'data/webqasim/Base5')

num_train_epochs = 2
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.9

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 32

# 预训练模型
train = True

# 测试模型
test = False

# 在线测试
online = True
