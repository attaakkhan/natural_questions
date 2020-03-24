
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


# NQ Models Section 
### We implemented two baseline model Bert and Decatt-Docreader which are under the file baseline


# Materials and Refrences Section
1) github.com/google-research-datasets/natural-questions
2) github.com/google-research/language/tree/master/language/question_answering
3) http://jalammar.github.io/illustrated-transformer/
4) http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time


