
# CSE 576 NLP Group | NQ

#### Members
###### Zhaomeng Wang, Trenton Gailey, Zahra Zahedi, Zheyin Liang, Atta Khan


# Note:
#### At this point we used two baseline models, Bert and Deccat-Docreader
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


### Training bertjoint--Using the Dev Set
### prepare data--convert train to Tf Records
```
$ python -m language.question_answering.bert_joint.prepare_nq_data   --logtostderr   --input_jsonl data/dev/nq-dev-??.jsonl.gz   --output_tfrecord bert-joint-baseline/???   --max_seq_length=512   --include_unknowns=0.02   --vocab_file=bert-joint-baseline/vocab-nq.txt


```

### Output
```
WARNING: Logging before flag parsing goes to stderr.
W0326 03:23:11.562084 139854894638976 module_wrapper.py:139] From /usr/local/lib/python2.7/dist-packages/bert/optimization.py:87: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

W0326 03:23:13.444865 139854894638976 module_wrapper.py:139] From /usr/local/lib/python2.7/dist-packages/bert/tokenization.py:125: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

I0326 03:23:14.498956 139854894638976 prepare_nq_data.py:75] Examples processed: 0
I0326 03:23:54.666284 139854894638976 prepare_nq_data.py:75] Examples processed: 100
I0326 03:24:25.081033 139854894638976 prepare_nq_data.py:75] Examples processed: 200
I0326 03:24:57.720694 139854894638976 prepare_nq_data.py:75] Examples processed: 300
I0326 03:25:30.721096 139854894638976 prepare_nq_data.py:75] Examples processed: 400
I0326 03:26:09.302308 139854894638976 prepare_nq_data.py:75] Examples processed: 500
I0326 03:26:40.885102 139854894638976 prepare_nq_data.py:75] Examples processed: 600
I0326 03:27:13.591331 139854894638976 prepare_nq_data.py:75] Examples processed: 700
I0326 03:27:44.374589 139854894638976 prepare_nq_data.py:75] Examples processed: 800
I0326 03:28:18.975951 139854894638976 prepare_nq_data.py:75] Examples processed: 900
I0326 03:28:53.517859 139854894638976 prepare_nq_data.py:75] Examples processed: 1000
I0326 03:29:31.464816 139854894638976 prepare_nq_data.py:75] Examples processed: 1100
I0326 03:30:07.438111 139854894638976 prepare_nq_data.py:75] Examples processed: 1200
I0326 03:30:40.177573 139854894638976 prepare_nq_data.py:75] Examples processed: 1300
I0326 03:31:09.506659 139854894638976 prepare_nq_data.py:75] Examples processed: 1400
I0326 03:31:45.806902 139854894638976 prepare_nq_data.py:75] Examples processed: 1500
I0326 03:32:21.154177 139854894638976 prepare_nq_data.py:80] Examples with correct context retained: 1545 of 1600

```
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

 
   
   
   
   
# Materials and Refrences Section
1) github.com/google-research-datasets/natural-questions
2) github.com/google-research/language/tree/master/language/question_answering
3) http://jalammar.github.io/illustrated-transformer/
4) http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time


