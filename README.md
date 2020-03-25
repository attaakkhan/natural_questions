
# CSE 576 NLP Group | NQ

#### Members
###### Zhaomeng Wang, Trenton Gailey, Zahra Zahedi, Zheyin Liang, Atta Khan


# Note:
#### At this point we used two baseline models, Bert and Deccat-Docreader


# Testing Section


### Classification with already trained bert

Testing already trained bert and classifying sentences

```
$  pip3 install numpy pandas torch transformers sklearn
$  python3.6 do_classification_with_already_trained_bert.py
```



# Datasets Section

```
$  pip install gsutil
$  gsutil -m cp -r gs://natural_question data

```
### DownLoad The Dev Set 4 files--(1.0 GB)(3-TRAIN, 1 TEST)

$ cd /natural_questions/baseline/language/language/question_answering/bert_joint
$ mkdir data
$ /usr/lib/google-cloud-sdk/bin/gsutil -m cp -R gs://natural_questions/v1.0/dev ./data





# NQ Models Section 
### We implemented two baseline model Bert and Decatt-Docreader which are under the file baseline
## Bert-Joint

### Pre-req
```
$ sudo -H pip install --upgrade pip
$ pip install bert-tensorflow natural-questions
$ git clone https://github.com/attaakkhan/natural_questions.git
$ cd natural_questions/baseline/language
$ pip install -r requirements.txt
$ cd language/question_answering/bert_joint
```



### Download the preprocessed bert

```
$ gsutil cp -R gs://bert-nq/bert-joint-baseline
```


### Training bertjoint
TODO


### Prediction
```
$ python -m run_nq \
   --logtostderr \
   --bert_config_file=bert-joint-baseline/bert_config.json \
   --vocab_file=bert-joint-baseline/vocab-nq.txt \
   --predict_file=tiny-dev/nq-dev-sample.no-annot.jsonl.gz \
   --init_checkpoint=bert-joint-baseline/bert_joint.ckpt \
   --do_predict \
   --output_dir=bert_model_output \
   --output_prediction_file=bert_model_output/predictions.json
  ```
### Evalvation
TODO
 
   
   
   
   
# Materials and Refrences Section
1) github.com/google-research-datasets/natural-questions
2) github.com/google-research/language/tree/master/language/question_answering
3) http://jalammar.github.io/illustrated-transformer/
4) http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time


