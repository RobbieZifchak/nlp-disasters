import pandas as pd
pd.set_option('display.max_rows', 999)
import nltk
import matplotlib.pyplot as plt
import matplotlib as cm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import string, re
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('train_with_tokens.csv')
testdf = pd.read_csv('test_with_tokens.csv')



train_list = df.token.tolist()
test_list = testdf.token.tolist()
y = df.target.tolist()


train_X = []
for word in train_list:
    joined = ''.join(x for x in word)
    train_X.append(joined)


test_X = []
for word in test_list:
    joined = ''.join(x for x in word)
    test_X.append(joined)

test_X


X_train, X_test, y_train, y_test = train_test_split(train_X, y, test_size=0.20, random_state=1)

cv = CountVectorizer()
# word_count_vector = cv.fit_transform(train_X)
# word_count_vector.shape

X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)


tfidf = TfidfVectorizer()
tfidf_data_train = tfidf.fit_transform(X_train)
tfidf_data_test = tfidf.transform(X_test)




clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)
print(classification_report(y_test, y_predicted_counts))



# random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train_counts, y_train)
rf_train_preds = rf_classifier.predict(X_train_counts)
rf_test_preds = rf_classifier.predict(X_test_counts)
print(classification_report(y_test, rf_test_preds))
cm2 = confusion_matrix(y_test, rf_test_preds)
fig = plt.figure(figsize=(10, 10))
print(cm2)


import seaborn as sns

sns.heatmap(cm2.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Fake Disaster', 'Real Disaster'], yticklabels=['Fake Disaster', 'Real Disaster'])
bottom, top = plt.ylim()
plt.ylim(bottom + 0.5, top - 0.5)
plt.xlabel('true label')
plt.ylabel('predicted label');





# random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(tfidf_data_train, y_train)
rf_train_preds = rf_classifier.predict(tfidf_data_train)
rf_test_preds = rf_classifier.predict(tfidf_data_test)
print(classification_report(y_test, rf_test_preds))


cm2 = confusion_matrix(y_test, rf_test_preds)
fig = plt.figure(figsize=(10, 10))
print(cm2)


from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

sns.heatmap(cm2.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['Fake Disaster', 'Real Disaster'], yticklabels=['Fake Disaster', 'Real Disaster'])
bottom, top = plt.ylim()
plt.ylim(bottom + 0.5, top - 0.5)
plt.xlabel('true label')
plt.ylabel('predicted label');


# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, rf_train_preds)
test_accuracy = accuracy_score(y_test, rf_test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation Accuracy: {:.4}%'.format(test_accuracy * 100))








import xgboost

clf = xgboost.XGBClassifier()
clf.fit(X_train_counts, y_train)
training_preds = clf.predict(X_train_counts)
test_preds = clf.predict(X_test_counts)
print(classification_report(y_test, test_preds))

# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation Accuracy: {:.4}%'.format(test_accuracy * 100))
