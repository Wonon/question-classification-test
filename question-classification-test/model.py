import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report, plot_confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# for text cleaner
from stop_words import get_stop_words
import string
# download dependency
import nltk
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
): nltk.download(dependency)
# warnings
import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

def text_cleaner(text: str):
    """
    returns text in lower case without stop words, digits, or punctuation
    """
    cleaned = [word for word in text.split() if word not in get_stop_words('english')]
    cleaned = ' '.join([i for i in cleaned if not i.isdigit()])
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    return cleaned.lower()

# load data
data = pd.read_csv("data\FactSetData.tsv", sep='\t')
data["cleaned_question"] = data["question"].apply(text_cleaner)
# split into training and validation 
X = data["cleaned_question"]
y = data.Category.values
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# create a classifier using sklearn.pipeline
question_classifier = Pipeline(steps=[
    ('pre_processing',TfidfVectorizer(lowercase=False)),
    ('naive_bayes',MultinomialNB())
])
# train & test the question classifier
question_classifier.fit(X_train,y_train)
y_preds = question_classifier.predict(X_valid)
model_acc = accuracy_score(y_valid,y_preds)

# save the model 
import joblib 
joblib.dump(question_classifier, 'savedmodels\question_model_pipeline.pkl')