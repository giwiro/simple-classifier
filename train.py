import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dataset import load_dataset_pd, normalize_text, load_test_dataset_pd

df = load_dataset_pd()
df_train = load_test_dataset_pd()

# df["category_id"] = [map_categories(c.str) for c in df["category"]]

x_train = df["name"]
y_train = df["category"]

x_test = df_train["name"]
y_test = df_train["category"]

print("Creating CountVectorizer...")
# this calculates a vector of term frequencies for
# each document
c_vectorizer = CountVectorizer()

# this normalizes each term frequency by the
# number of documents having that term
tfidf = TfidfTransformer()

# this is a linear SVM classifier
clf = LinearSVC()

# this is logistic regression
lg = LogisticRegressionCV()

pipeline = Pipeline([
    ("vect", c_vectorizer),
    ("tfidf", tfidf),
    ("lg", lg)
])

# call fit as you would on any classifier
pipeline.fit(x_train, y_train)

# predict test instances
y_preds = pipeline.predict(x_test)

# calculate f1
# mean_f1 = f1_score(y_test, y_preds, average="micro")
y_preds_proba = pipeline.predict_proba(x_test)

print(x_test.size)

print(y_preds_proba)

print(classification_report(y_test, y_preds))

# print(mean_f1)
