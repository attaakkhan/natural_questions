#! /bin/bash
#SBATCH -n 32
#SBATCH --mem-per-cpu=32000
#SBATCH -J final_script
#SBATCH -o %j.out
#SBATCH -e %j.error
#SBATCH -t 0-8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zliang35@asu.edu

source python2/py2/bin/activate
python -m natural_questions.nq_eval \
  --logtostderr \
  --gold_path=tiny-dev/nq-dev-sample.jsonl.gz \
  --predictions_path=bert_model_output/predictions.json
