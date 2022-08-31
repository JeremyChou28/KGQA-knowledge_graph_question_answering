# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/6/28 14:15

"""
构造NER数据集，将序列进行实体标注，用于BERT+BiLSTM+CRF模型的NER训练
"""

import pandas as pd

for data_type in ["train", "test"]:
    if data_type == "train":
        file = "./webqa/train.csv"
    elif data_type == "test":
        file = "./webqa/test.csv"
    else:
        raise "error:the data_type should be train or test"

    q_t_a_list = []
    seq_q_list = []
    seq_tag_list = []

    df = pd.read_csv(file)

    for i in range(len(df)):
        q_str = df.loc[i]['question'].strip()
        entity = str(df.loc[i]['entity']).strip()
        if entity in q_str:
            q_list = q_str.split(' ')
            q_list[-1] = q_list[-1].split('?')[0]
            q_list.append('?')
            entities = entity.split(' ')
            seq_q_list.extend(q_list)
            seq_q_list.extend([" "])
            tag_list = ["O" for i in range(len(q_list))]
            try:
                item = entities[0]
                for i in range(len(q_list)):
                    if item in q_list[i]:
                        tag_list[i] = "B-LOC"
                        for j in range(1, len(entities)):
                            tag_list[i + j] = "I-LOC"
                        break
            except:
                print(entities, q_list)
            seq_tag_list.extend(tag_list)
            seq_tag_list.extend([" "])
        else:
            pass

    seq_result = [str(q) + " " + tag for q, tag in zip(seq_q_list, seq_tag_list)]
    print(len(seq_result))

    if data_type == "test":  # 12264
        with open("./webqaner/" + "test" + ".txt", "r+", encoding='utf-8') as f:
            f.write("\n".join(seq_result))
            f.close()
    elif data_type == "train":  # 23207
        with open("./webqaner/" + "train" + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join(seq_result[0:17736]))
            f.close()
        with open("./webqaner/" + "dev" + ".txt", "w", encoding='utf-8') as f:
            f.write("\n".join(seq_result[17736:21283]))
            f.close()
        with open("./webqaner/" + "test" + ".txt", "w+", encoding='utf-8') as f:
            f.write("\n".join(seq_result[21283:]))
            f.close()
    else:
        raise "error:the data_type should be train or test"
