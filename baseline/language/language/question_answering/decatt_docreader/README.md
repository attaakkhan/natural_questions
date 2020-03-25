# Get Started

### Create a directory for downloaded data:

```shell
mkdir -p data
```
### Download the NQ data:

```shell
pip install gsutil
gstuil -m cp -r gs://natural_questions data
```

### Preprocess data for the short answer pipeline model:
(You can change the training set or development set to whatever you want, the dataset included in the commands below are what I used to train in my PC)

```shell
python preprocessing/create_nq_short_pipeline_examples.py \
--input_pattern=data/natural_questions/v1.0/train/nq-train-06.jsonl.gz \
--output_dir=data/natural_questions/v1.0/sample/train

python preprocessing/create_nq_short_pipeline_examples.py \
--input_pattern=data/natural_questions/v1.0/dev/nq-dev-01.jsonl.gz \
--output_dir=data/natural_questions/v1.0/sample/dev
```

### Preprocess data for the long answer model:

```shell
python preprocessing/create_nq_long_examples.py \
--input_pattern=data//natural_questions/v1.0/sample/nq-train-sample.jsonl.gz \
--output_dir=data/natural_questions/v1.0/sample/train

python preprocessing/create_nq_long_examples.py \
--input_pattern=data//natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz \
--output_dir=data/natural_questions/v1.0/sample/dev 
```

### Download pre-trained word embeddings:
```shell
curl http://nlp.stanford.edu/data/glove.840B.300d.zip > data/glove.840B.300d.zip
unzip data/glove.840B.300d.zip -d data
```

### Train your own short answer pipeline model:
```shell
python experiments/nq_short_pipeline_experiment.py \
--embeddings_path=data/glove.840B.300d.txt \
--nq_short_pipeline_train_pattern=data/natural_questions/v1.0/sample/train/nq-train-06.short_pipeline.tfr \
--nq_short_pipeline_eval_pattern=data/natural_questions/v1.0/sample/dev/nq-dev-01.short_pipeline.tfr \
--num_eval_steps=10 --num_train_steps=50 --model_dir=models/nq_short_pipeline

```

### Train your own long answer model:
```shell
python experiments/nq_long_experiment.py \
--embeddings_path=data/glove.840B.300d.txt \
--nq_long_train_pattern=data/natural_questions/v1.0/sample/train/nq-train-sample.long.tfr \
--nq_long_eval_pattern=data/natural_questions/v1.0/sample/dev/nq-dev-sample.long.tfr \
--num_eval_steps=50 \
--batch_size=4 \
--num_train_steps=200 \
--model_dir=models/nq_long
```
The model of long answer is quite huge. So I did not upload it. And the prediction accuracy of it on sample dev-set of NQ is 50%.

### Example output of short answer pipeline model:
[Google Drive](https://drive.google.com/drive/folders/1mUIgolfLt6c2_0ffkiHI80gMoWW8lODh?usp=sharing)
