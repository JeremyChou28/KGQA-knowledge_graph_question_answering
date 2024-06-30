# KGQA：question answering based on knowledge graph

## Introduction

本项目主要包含两个重要的模块：一是实体识别模块，二是关系预测/语义相似度计算模块。

实体识别模块是利用**BERT+Softmax/BERT+CRF/BERT+BiLSTM+CRF模型**实现

关系预测/语义相似度计算模块是基于**BERT+Softmax模型进行句子对二分类任务**实现

## 环境配置

    Python版本为3.7
    tensorflow版本为1.13
    Navicat Premium12
    python其他包的安装：
    pip install -r requirements.txt

## 目录说明

    .
    └─QA_basedKG
      ├─README.md
      ├─bert							google官方下载的BERT模型
      ├─data							数据集和处理后的数据
        ├─nlpcc2016    					NLPCC 2016 KBQA 原始数据集和修正数据集
        ├─nlpcc2016ner      			NLPCC 2016 KBQA 修正后的实体识别模块数据集
        ├─nlpcc2016sim              	NLPCC 2016 KBQA 语义相似度模块数据集和代码
        ├─originNLPCC2016ner        	NLPCC 2016 KBQA 原始的实体识别模块数据集
        ├─webqa                   		WebQuestions数据集
        ├─webqaner               		WebQuestions实体识别模块数据集
        ├─webqasim                  	WebQuestions语义相似度模块数据集
        ├─construct_nlpcc2016ner.py     构建NLPCC 2016 KBQA 实体识别模块数据集代码
        ├─construct_webqaner.py         构建WebQuestions 实体识别模块数据集代码
        ├─nlpcc2016_triple_clean.py		修正NLPCC 2016 KBQA数据集代码
        ├─load_dbdata.py             	存入数据库
      ├─src
        ├─args.py               				语义相似度模块超参数设置
        ├─bert_bilstm_crf.py            		bert-bilstm-crf模型代码
        ├─bert_crf.py                   		bert-crf模型代码
        ├─bert_softmax.py               		bert-softmax模型代码        
        ├─bert_bilstm_crf_ner_predict.py        bert-bilstm-crf模型nlpcc2016kbqa实体预测代码                   
        ├─bert_bilstm_crf_webqa_predict.py      bert-bilstm-crf模型webqa实体预测代码            
        ├─bert_crf_ner_predict.py               bert-crf模型nlpcc2016kbqa实体预测代码     
        ├─bert_crf_webqa_predict.py             bert-crf模型webqa实体预测代码        
        ├─bert_softmax_ner_predict.py           bert-softmax模型nlpcc2016kbqa实体预测代码         
        ├─bert_softmax_webqa_predict.py         bert-softmax模型webqa实体预测代码             
        ├─conlleval.py                          输出实体预测结果
        ├─conlleval.pl                         
        ├─global_config.py						日志打印函数
        ├─kbqa_nlpcc2016.py						结合语义匹配和非语义匹配的nlpcc2016kbqa问答代码
        ├─kbqa_nlpcc2016_semantic.py  			纯语义匹配的nlpcc2016kbqa问答代码
        ├─kbqa_online.py						在线kbqa问答代码
        ├─kbqa_webqa.py							纯语义匹配的webqa的问答代码
        ├─lstm_crf_layer.py						lstm-crf模型代码
        ├─run_similarity.py						语义相似度模块代码
        ├─tf_metrics.py							评价指标
      ├─nerpredict                             	存放实体预测后的文件
      ├─log                             		存放代码运行日志
      ├─checkpoints                            	存放模型训练结果
      ├─scripts                        			存放运行脚本
      ├─requirements.txt                       	项目运行环境包
      ├─ModelParams								存放下载的BERT的预训练模型文件

## 快速入门（以NLPCC 2016 KBQA数据集为例）

```python
1.数据预处理，构建数据集
运行 data/construct_nlpcc2016ner.py生成nlpcc2016ner/train.txt,dev.txt,test.txt，即实体识别模块的训练集、验证集和测试集
运行 data/load_dbdata.py将数据集中的知识三元组存入mysql数据库
运行 data/nlpcc2016sim/Base3.py等文件生成语义相似度模块的训练集、验证集和测试集


2.NER训练
bash nlpcc2016_bert_bilstm_crf.sh  		采用bert-bilstm-crf模型来做实体识别任务
bash nlpcc2016_bert_crf.sh  			采用bert-crf模型来做实体识别任务
bash nlpcc2016_bert_softmax.sh  		采用bert-softmax模型来做实体识别任务

3.NER预测
bash bert_bilstm_crf_ner_predict.sh 	bert-bilstm-crf做实体预测
bash bert_crf_ner_predict.sh  			bert-crf做实体预测
bash bert_softmax_ner_predict.sh  		bert-softmax做实体预测

4.语义相似度模块训练
args.py用来修改语义相似度模块的超参数
bash run_similarity.sh

5.KBQA问答
python kbqa_nlpcc2016_semantic.py
```

​    



