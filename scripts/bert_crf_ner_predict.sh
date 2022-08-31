cd ..
cd src
nohup python bert_crf_ner_predict.py \
  --task_name=ner \
  --data_dir=../data/nlpcc2016ner \
  --vocab_file=../ModelParams/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=../ModelParams/chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=../checkpoints/nlpcc2016/ner/bert_crf \
  --init_checkpoint=../ModelParams/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --data_config_path=../Config/nlpcc2016/bert_crf_ner.conf \
  --do_train=True \
  --do_eval=True \
  --max_seq_length=128 \
  --train_batch_size=64 \
  --eval_batch_size=8 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=1 \
  --do_predict_online=False \
  --do_predict_outline=True > ../log/nlpcc2016/ner/bert_crf_ner_predict.log 2>&1 &
jobs -l