import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

batch_1 = df[:2000]
batch_1[1].value_counts()
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# only x and z axis

features = last_hidden_states[0][:,0,:].numpy()
labels = batch_1[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

#parameters = {'C': np.linspace(0.0001, 100, 20)}
#grid_search = GridSearchCV(LogisticRegression(), parameters)
#grid_search.fit(train_features, train_labels)

scaler = preprocessing.StandardScaler().fit(train_features)
train_featuresS=scaler.transform(train_features)
test_featuresS=scaler.transform(test_features)


# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(train_featuresS, train_labels)

print(lr_clf.score(test_featuresS, test_labels))
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_featuresS, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


