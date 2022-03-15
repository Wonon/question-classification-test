# uvicorn main:app --reload
# .../docs

from fastapi import FastAPI 
from stop_words import get_stop_words
import string
from os.path import dirname, join, realpath
import joblib
from model import model_acc

app = FastAPI(
    title = "Question Classification Test API",
    description = "API that classifies the intent of a question using a NLP model.",
    version = "0.1"
)

def text_cleaner(text: str):
    """
    returns text in lower case without stop words, digits, or punctuation
    """
    cleaned = [word for word in text.split() if word not in get_stop_words('english')]
    cleaned = ' '.join([i for i in cleaned if not i.isdigit()])
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    return cleaned.lower()

# load the question classifier model
with open(
    join(dirname(realpath(__file__)), "savedmodels\question_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

# api endpoint
@app.get("/question-classifier")
def process(question: str):
    """
    receives the user's question and predicts the question of its content
    """
    # clean question
    cleaned_question = text_cleaner(question)
    # perform prediction
    prediction = model.predict([cleaned_question])
    output = (prediction[0])
    probas = model.predict_proba([cleaned_question])
    probas_dict = dict(zip(list((model.classes_)), probas.tolist()[0]))
    result = {"Model Accuracy": str(model_acc), "Prediction": output, "Distribution": probas_dict}
    return result