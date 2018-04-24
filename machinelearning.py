import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text import TfidfVectorizer
from timeit import default_timer as timer

wine = pd.read_csv('wine.csv')

confmatrix = np.reshape(('   neg   ', 'false pos', 'false neg', '   pos   '), (2,2))

NBresult = 0
NB_AUCscores = []
NBmatrix = np.zeros((2,2))

#SVMresult = 0
SVM_AUCscores = []
SVMmatrix = np.zeros((2,2))

#LRresult = 0
LR_AUCscores = []
LRmatrix = np.zeros((2,2))

kf = StratifiedKFold(n_splits=10)

start = timer()
for train_index, test_index in kf.split(wine.desc, wine.binary_label):
    X_train, X_test = wine.desc[train_index], wine.desc[test_index]
    y_train, y_test = wine.binary_label[train_index], wine.binary_label[test_index]
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8) #min_df=5, max_df = 0.8, 
    train_tfidf = vectorizer.fit_transform(X_train)
    test_tfidf = vectorizer.transform(X_test)
    
    NBmodel = MultinomialNB()
    NBmodel.fit(train_tfidf, y_train)
    NBresult = NBmodel.predict(test_tfidf)
    NBmatrix += confusion_matrix(y_test, NBresult)
    NB_AUCscores.append(roc_auc_score(y_test, NBresult))
    
    SVMmodel = LinearSVC()
    SVMmodel.fit(train_tfidf, y_train)
    SVMresult = SVMmodel.predict(test_tfidf)
    SVMmatrix += confusion_matrix(y_test, SVMresult)
    SVM_AUCscores.append(roc_auc_score(y_test, SVMresult))
    
end = timer()
    
NBmatrix /= len(wine)
NBscore = round(NBmatrix[0,0] + NBmatrix[1,1], 3)
NBauc = round(np.mean(NB_AUCscores), 3)
print('NB\nScore: ' + str(NBscore) + '  AUC:' + str(NBauc))

SVMmatrix /= len(wine)
SVMscore = round(SVMmatrix[0,0] + SVMmatrix[1,1], 3)
SVMauc = round(np.mean(SVM_AUCscores), 3)
print('SVM\nScore: ' + str(SVMscore) + '  AUC:' + str(SVMauc))

print((end - start) / 60)
