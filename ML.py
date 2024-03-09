from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class model:
    def __init__(self):
        self.df = pd.read_csv('Cleaned_Data.csv')
        self.df['Email'] = self.df.Email.apply(lambda email: np.str_(email))
        self.Data = self.df.Email
        self.Labels = self.df.Label
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(self.Data,self.Labels,random_state=10)
        self.training_data_list = self.training_data.to_list()
        self.vectorizer = TfidfVectorizer()
        self.training_vectors = self.vectorizer.fit_transform(self.training_data_list)
        self.model_nb = MultinomialNB()
        self.model_svm = SVC(probability=True)
        self.model_lr = LogisticRegression()
        self.model_knn = KNeighborsClassifier(n_neighbors=9)
        self.model_rf = RandomForestClassifier(n_estimators=19)
        self.model_nb.fit(self.training_vectors, self.training_labels)
        self.model_lr.fit(self.training_vectors, self.training_labels)
        self.model_rf.fit(self.training_vectors, self.training_labels)
        self.model_knn.fit(self.training_vectors, self.training_labels)
        self.model_svm.fit(self.training_vectors, self.training_labels)
    def get_prediction(self,vector):
        pred_nb=self.model_nb.predict(vector)[0]
        pred_lr=self.model_lr.predict(vector)[0]
        pred_rf=self.model_rf.predict(vector)[0]
        pred_svm=self.model_svm.predict(vector)[0]
        pred_knn=self.model_knn.predict(vector)[0]
        preds=[pred_nb,pred_lr,pred_rf,pred_svm,pred_knn]
        spam_counts=preds.count(1)
        if spam_counts>=3:
            return 'Spam'
        return 'Non-Spam'
    def get_probabilities(self,vector):
        prob_nb=self.model_nb.predict_proba(vector)[0]*100
        prob_lr = self.model_lr.predict_proba(vector)[0] * 100
        prob_rf = self.model_rf.predict_proba(vector)[0] * 100
        prob_knn = self.model_knn.predict_proba(vector)[0] * 100
        prob_svm = self.model_svm.predict_proba(vector)[0] * 100
        return [prob_nb,prob_lr,prob_rf,prob_knn,prob_svm]

    def get_vector(self,text):
        return self.vectorizer.transform([text])

