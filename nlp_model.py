# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# text preprocessing
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# nltk downloads (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# =========================
# LOAD DATA
# =========================
df_train = pd.read_csv('train.csv')
print(df_train.shape)
print(df_train.head())

# =========================
# EDA
# =========================
sns.countplot(x='target', data=df_train)
plt.show()

# Feature engineering
df_train['word_count'] = df_train['text'].astype(str).apply(lambda x: len(x.split()))
df_train['char_count'] = df_train['text'].astype(str).apply(len)
df_train['unique_word_count'] = df_train['text'].astype(str).apply(lambda x: len(set(x.split())))

# =========================
# TEXT PREPROCESSING
# =========================
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess(text):
    text = str(text).lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def stopword_removal(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def stemming(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

# Apply preprocessing
df_train['clean_text'] = df_train['text'].apply(preprocess)
df_train['clean_text'] = df_train['clean_text'].apply(stopword_removal)
df_train['clean_text'] = df_train['clean_text'].apply(stemming)

# =========================
# TRAIN TEST SPLIT
# =========================
X = df_train['clean_text']
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TF-IDF VECTORIZATION
# =========================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =========================
# MODEL 1: LOGISTIC REGRESSION
# =========================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)

print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))

# =========================
# MODEL 2: NAIVE BAYES
# =========================
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)

print("Naive Bayes Classification Report")
print(classification_report(y_test, y_pred_nb))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# =========================
# ROC CURVE
# =========================
y_prob = lr.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
