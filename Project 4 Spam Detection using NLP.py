#!/usr/bin/env python
# coding: utf-8

# In[72]:


#Code to Read and Preview SMS Spam Collection Data 
name=r"C:\Users\istm\OneDrive - Ramboll\Desktop\ISHANT\Computer science\quant\linkedin#\NLL\Ex_Files_NLP_Python_ML_EssT\Ex_Files_NLP_Python_ML_EssT\Exercise Files\Ch01\01_03\Start\SMSSpamCollection.tsv"
rawdata=open(name).read()
rawdata[0:500]


# In[54]:


#Loading and Preparing SMS Spam Collection Data for NLP Processing
import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv(name, sep='\t')
data = pd.read_csv(name, sep='\t')
data.columns = ['label', 'body_text']


# In[55]:


data.head()


# In[56]:


#Text Cleaning Function for NLP Preprocessing
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords] # same as count vector unlike n gram vector.
    return text


# In[59]:


# Creating Feature number 1 that is Text Lengths for SMS Spam Detection
data['body_len']=data['body_text'].apply(lambda x:len(x)-x.count(" ") ) # remove white spces
data.head()


# In[60]:


# Creating Feature number 2 that is number of punctuations for SMS Spam Detection
import string 
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    # returning 1 if the char is punct there for using (in) not (not in) 
    return round(count/(len(text)-text.count(" ")),3)*100
    
data['punct %']=data['body_text'].apply(lambda x:count_punct(x))
data.head()


# In[61]:


#Visualizing SMS Length Distribution for Spam vs. Ham Messages
from matplotlib import pyplot
import numpy as np
bins=np.linspace(0,200,40) 
pyplot.hist(data[data['label']=='spam']['body_len'],bins,alpha=0.5,density = True,label='spam') 
#ham histo will dwarf the spam one if we not normalize as one is alot more in number compared to other 
pyplot.hist(data[data['label']=='ham']['body_len'],bins,alpha=0.5,density = True,label='ham')
pyplot.legend(loc='upper left')
pyplot.show() # spam appeared to be longer than regular text messages


# In[62]:


#Visualizing Percentage punctuation Distribution for Spam vs. Ham Messages
bins=np.linspace(0,50,40) 
pyplot.hist(data[data['label']=='spam']['punct %'],bins,alpha=0.5,density = True,label='spam') 
#ham histo will dwarf the spam one if we not normalize as one is alot more in number compared to other 
pyplot.hist(data[data['label']=='ham']['punct %'],bins,alpha=0.5,density = True,label='ham')
pyplot.legend(loc='upper right')
pyplot.show() # spam appeared to be longer than regular text messages


# In[63]:


#Creating TF-IDF Features and Combining with Text Attributes for the whole dataset for Grid Search purpose
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string

tfidf_vect = TfidfVectorizer(analyzer=clean_text) # weight of the words
X_tfidf = tfidf_vect.fit_transform(data['body_text'])

X_features = pd.concat([data['body_len'], data['punct %'], pd.DataFrame(X_tfidf.toarray())], axis=1)
X_features.head() 


# In[73]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[66]:


#Splitting Data into Training and Testing Sets
X_train,X_test,y_train,y_test=train_test_split(X_features,data['label'],test_size=0.2)


# In[67]:


#Training a Random Forest Classifier and Evaluating Performance
def train_RF(n_est,depth):
    rf=RandomForestClassifier(n_estimators=n_est,max_depth=depth,n_jobs=-1)
    rf_model=rf.fit(X_train,y_train)
    y_pred=rf_model.predict(X_test)
    precision,recall,fscore,support=score(y_test,y_pred,pos_label='spam',average='binary')
    print('Est:{}/Depth{}---Precisooon: {} / Recall:{} /Accuracy: {}'.format(n_est,depth,round(precision,3),round(recall,3),
                                                                            round(y_pred==y_test).sum()/len(y_pred)))


# In[71]:


# Performing manual Grid Search for Random Forest
for n_est in [50,100,150]:
    for depth in [10,20,30,None]:
        train_RF(n_est,depth)


# In[69]:


#Training a Gradient booster Classifier and Evaluating Performance
def train_GB(est, max_depth, lr):
    gb = GradientBoostingClassifier(n_estimators=est, max_depth=max_depth, learning_rate=lr)
    gb_model = gb.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
    print('Est: {} / Depth: {} / LR: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
        est, max_depth, lr, round(precision, 3), round(recall, 3), 
        round((y_pred==y_test).sum()/len(y_pred), 3)))


# In[70]:


# Performing manual Grid Search for Gradient booster
for n_est in [50, 100, 150]:
    for max_depth in [3, 7, 11, 15]:
        for lr in [0.01, 0.1, 1]:
            train_GB(n_est, max_depth, lr)


# In[ ]:


#Creating TF-IDF Features and Combining with Text Attributes for the training dataset.
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text']) 

tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text']) 

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True),  
           pd.DataFrame(tfidf_train.toarray())], axis=1) 
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1) 

X_train_vect.head()


# In[31]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time 


# In[32]:


#Training and Evaluating Random Forest Classifier Performance using the parameters giving the best results in Grid Search
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1) 

start = time.time()
rf_model = rf.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start) # no of sec to fit

start = time.time()
y_pred = rf_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))


# In[33]:


#Training and Evaluating Gradient Booster Classifier Performance using the parameters giving the best results in Grid Search
gb = GradientBoostingClassifier(n_estimators=150, max_depth=11) 

start = time.time()
gb_model = gb.fit(X_train_vect, y_train)
end = time.time()
fit_time = (end - start)

start = time.time()
y_pred = gb_model.predict(X_test_vect)
end = time.time()
pred_time = (end - start)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))


# - The Gradient Booster took 75 times longer to fit compared to the Random Forest, but the prediction times were almost identical.
# - The Random Forest achieved 100% precision but 81% recall, while the Gradient Booster had 92% precision and 82% recall.
# - It is advisable to use the Random Forest when false positives are costly, and the Gradient Booster when false negatives are more expensive.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




