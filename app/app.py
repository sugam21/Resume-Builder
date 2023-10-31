from flask import Flask , request, jsonify
import numpy as np
import pandas as pd
import re
import gradio
import pickle
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors # For loading saved gensim model
from simpletransformers.question_answering import QuestionAnsweringModel # This is for model

app = Flask(__name__)
@app.route("/")
def home():
    return "Hello World"
# you can get input using 2 methods GET(input through URL) and POST(Input without URL)
@app.route("/predict", methods = ["POST"])
def predict():
    # Write the things I will recieve through request
    question = request.form.get("question")

    to_predict =  [
    {
        "context": context,
        "qas": [
            {
                "question": question,
                "id": "0",
            }
        ],
    }]
    return jsonify(to_predict)


if __name__=="__main__":
    app.run(debug=True)