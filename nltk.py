#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Restaurant_Reviews.tsv',quoting=3,delimiter='\t')
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    sp=PorterStemmer()
    review=[sp.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
y=data.iloc[:,1].values
x=cv.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
    
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
    