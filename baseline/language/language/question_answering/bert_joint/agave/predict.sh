#! /bin/bash
#SBATCH -n 32
#SBATCH --mem-per-cpu=16000
#SBATCH -J final_script
#SBATCH -o %j.out
#SBATCH -e %j.error
#SBATCH -t 0-1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zliang35@asu.edu

source python2/py2/bin/activate
python run_nq.py \
  --logtostderr \
  --bert_config_file=bert-joint-baseline/bert_config.json \
  --vocab_file=bert-joint-baseline/vocab-nq.txt \
  --predict_file=tiny-dev/nq-dev-sample.no-annot.jsonl.gz \
  --init_checkpoint=bert_model_output/model.ckpt-0 \
  --do_predict \
  --output_dir=bert_model_output \
  --output_prediction_file=bert_model_output/predictions.json
