
# CSE 576 NLP Group | NQ

#### Members
   1) Zhaomeng Wang
   2) Trenton Gailey
   3) Zahra Zahedi
   4) Zheyin Liang
   5) Atta Khan



## Ubuntu
$ sudo apt-get install unzip

# Testing Section


### Classification with already trained bert

Testing bert and classifying sentences

```
$  pip3 install numpy pandas torch transformers sklearn
$  python3.6 do_classification_with_already_trained_bert.py
```


# Clone
```
$ git clone https://github.com/attaakkhan/natural_questions.git
```


# Datasets Section



### DownLoad The full dataset-- 50 train Files, 10 Dev files-- 307K NQ-train, 10k NQ-DEV
```
$ mkdir natural_questions/baseline/language/data
$ gsutil -m cp -R gs://natural_questions/v1.0 ./data
```




# NQ Models Section 

### We are testing and extending the current Bert-Joint-- avialable online on github, in the Google Language repo.

### Pre-req
```
$ sudo -H pip install --upgrade pip
$ pip install bert-tensorflow natural-questions
$ cd natural_questions/baseline/language
$ pip install -r requirements.txt
$ python setup.py install

```



### Download the preprocessed joint_bert

```
$ gsutil cp -R gs://bert-nq/bert-joint-baseline
```


### Training bertjoint--Using the tf_records
### repare data--convert train to tf_records
```
$ python -m language.question_answering.bert_joint.prepare_nq_data   --logtostderr   --input_jsonl data/dev/nq-dev-??.jsonl.gz   --output_tfrecord bert-joint-baseline/???   --max_seq_length=512   --include_unknowns=0.02   --vocab_file=bert-joint-baseline/vocab-nq.txt


```

### Output

### download the pretrained bert-- Bert Uncased
```
$ wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
$ unzip uncased_L-24_H-1024_A-16.zip 
```
### Train the model-Using the TF Record
```
$  python -m language.question_answering.bert_joint.run_nq   --logtostderr   --bert_config_file=bert-joint-baseline/bert_config.json   --vocab_file=bert-joint-baseline/vocab-nq.txt   --train_precomputed_file=bert-joint-baseline/nq-dev.tfrecords-00000-of-00001   --train_num_precomputed=494670   --learning_rate=3e-5   --num_train_epochs=1   --max_seq_length=512   --save_checkpoints_steps=5000   --init_checkpoint=uncased_L-24_H-1024_A-16/bert_model.ckpt   --do_train   --output_dir=bert_model_output
```

## Testing using the the tiny-dev


### Prediction
```
$ python -m language.question_answering.bert_joint.run_nq \
   --logtostderr \
   --bert_config_file=bert-joint-baseline/bert_config.json \
   --vocab_file=bert-joint-baseline/vocab-nq.txt \
   --predict_file=tiny-dev/nq-dev-sample.gz \
   --init_checkpoint=bert-joint-baseline/bert_joint.ckpt \
   --do_predict \
   --output_dir=bert_model_output \
   --output_prediction_file=bert_model_output/predictions.json
  ```
### Evalvation

  ```
$ python -m natural_questions.nq_eval \
   --logtostderr \
   --gold_path=tiny-dev/nq-dev-sample.jsonl.gz \
   --predictions_path=bert_model_output/predictions.json
   
   

  ```
# Results
### M1: Orignal Results

```
    "long-best-threshold-f1": 0.6168224299065421,
    "long-best-threshold-precision": 0.5945945945945946,
    "long-best-threshold-recall": 0.6407766990291263,
    "short-best-threshold-f1": 0.5619834710743801,
    "short-best-threshold-precision": 0.7391304347826086,
    "short-best-threshold-recall": 0.4533333333333333
```

### M2:Results after applying  post processing to M1. 
```
    "long-best-threshold-f1": 0.6407766990291263,
    "long-best-threshold-precision": 0.6407766990291263,
    "long-best-threshold-recall": 0.6407766990291263,
    "short-best-threshold-f1": 0.5522388059701493,
    "short-best-threshold-precision": 0.6271186440677966,
    "short-best-threshold-recall": 0.49333333333333335,
```

### M3: Results after applying max_contexts=120 to M1
```
    "long-best-threshold-f1": 0.6355140186915887,
    "long-best-threshold-precision": 0.6126126126126126,
    "long-best-threshold-recall": 0.6601941747572816,
    "short-best-threshold-f1": 0.5619834710743801,
    "short-best-threshold-precision": 0.7391304347826086,
    "short-best-threshold-recall": 0.4533333333333333,
 ```


### M4: Combining M2 and M3
```
    "long-best-threshold-f1": 0.6513761467889908,
    "long-best-threshold-precision": 0.6173913043478261,
    "long-best-threshold-recall": 0.6893203883495146,
    "short-best-threshold-f1": 0.5606060606060606,
    "short-best-threshold-precision": 0.6491228070175439,
    "short-best-threshold-recall": 0.49333333333333335,
```

### M5: fined tuned wwm_uncased_bert on NQ
```
    "long-best-threshold-f1": 0.6494845360824741,
    "long-best-threshold-precision": 0.6923076923076923,
    "long-best-threshold-recall": 0.6116504854368932,
    "short-best-threshold": 7.520410180091858,
    "short-best-threshold-f1": 0.5644171779141104,
    "short-best-threshold-precision": 0.5227272727272727,
    "short-best-threshold-recall": 0.6133333333333333,
```

### M6: Combining M5 and M4
```
    "long-best-threshold-f1": 0.6733668341708543,
    "long-best-threshold-precision": 0.6979166666666666,
    "long-best-threshold-recall": 0.6504854368932039,
    "short-best-threshold": 8.220512390136719,
    "short-best-threshold-f1": 0.5850340136054423,
    "short-best-threshold-precision": 0.5972222222222222,
    "short-best-threshold-recall": 0.5733333333333334,
```



 
   
   
   
   
# Materials and Refrences Section
1) github.com/google-research-datasets/natural-questions
2) github.com/google-research/language/tree/master/language/question_answering
3) http://jalammar.github.io/illustrated-transformer/
4) http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time


