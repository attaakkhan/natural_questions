#! /bin/bash
#SBATCH -n 32
#SBATCH --mem-per-cpu=64000
#SBATCH -J final_script
#SBATCH -o %j.out
#SBATCH -e %j.error
#SBATCH -t 0-48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zliang35@asu.edu

source python2/py2/bin/activate
python run_nq.py \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --train_precomputed_file=bert-joint-baseline/nq-train.tfrecords-00000-of-00001 \
  --train_num_precomputed=494670 \
  --learning_rate=3e-5 \
  --num_train_epochs=1 \
  --max_seq_length=512 \
  --save_checkpoints_steps=500 \
  --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt \
  --do_train \
  --output_dir=bert_model_output
