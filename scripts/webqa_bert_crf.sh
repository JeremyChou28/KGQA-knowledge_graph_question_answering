cd ..
cd src
nohup python bert_crf.py \
  --task_name=ner \
  --data_dir=../data/webqaner \
  --vocab_file=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/bert_config.json \
  --output_dir=../checkpoints/webqa/ner/bert_crf \
  --init_checkpoint=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --data_config_path=../Config/webqa/bert_crf_ner.conf \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=1 \
  --warmup_proportion=0.1 \
  --crf=True > ../log/webqa/ner/bert_crf.log 2>&1 &

jobs -l
