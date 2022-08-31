cd ..
cd src
nohup python bert_crf_webqa_predict.py \
  --task_name=ner \
  --data_dir=../data/webqaner \
  --vocab_file=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/bert_config.json \
  --output_dir=../checkpoints/webqa/ner/bert_crf \
  --init_checkpoint=/home/zhoujianping/bert-chinese-ner/checkpoint/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --data_config_path=../Config/webqa/bert_crf_ner.conf \
  --do_train=True \
  --do_eval=True \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=1 \
  --do_predict_online=False \
  --do_predict_outline=True > ../log/webqa/ner/bert_crf_webqa_predict.log 2>&1 &
jobs -l