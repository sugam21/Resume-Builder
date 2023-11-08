from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pandas as pd
import random
import re
from typing import List
import time
import pickle
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors  # For loading saved gensim model
from simpletransformers.question_answering import (
    QuestionAnsweringModel,
)  # This is for model

# --------------------------------------------------------- BEFORE HAND CLASSES AND FUNCTIONS ----------------------------

#            ➡️➡️➡️ Stores the required directories
# ----------------------------------------------------------------
# ----------------------------------------------------------------


model_dir_ = "/home/sugam/Work/10-19 NLP/12 Projects/Resume Builder/Output/"
data_dir = "/home/sugam/Work/10-19 NLP/12 Projects/Resume Builder/data/Processed/"

model = model_dir_ + "bert/"
context_data = data_dir + "context_data.csv"


# ----------------------------------------------------------------
# ➡️➡️➡️➡️➡️ PREPROCESSING PIPELINE
# ----------------------------------------------------------------
# ----------------------------------------------------------------


class LowerCasing(BaseEstimator, TransformerMixin):
    """Takes the string and converts into lower casing"""

    def fit(self, text, y=None):
        return self

    def transform(self, text):
        return text.lower()


class RemovePunctuation(BaseEstimator, TransformerMixin):
    """Takes the string and removes punctuation"""

    def fit(self, text, y=None):
        return self

    def transform(self, text):
        exclude = "!\"#$%&'()*+./:;<=>?@[\\]^`{|}~"
        text = text.translate(str.maketrans("", "", exclude))
        text = re.sub(",", " ", text)
        text = re.sub(r"\(", " ", text)
        text = re.sub(r"\)", " ", text)

        return text


class RemoveAccent(BaseEstimator, TransformerMixin):
    """Takes string and removes accent words"""

    def fit(self, text, y=None):
        return self

    def transform(self, text):
        accent_letters = "éàáñüãèìöäøõîûçôšâ"
        text = text.translate(str.maketrans("", "", accent_letters))

        return text


class RemoveStopWords(BaseEstimator, TransformerMixin):
    """Takes the string and remove the stopwords"""

    def fit(self, text, y=None):
        return self

    def transform(self, text):
        new_text = []
        for words in text.split():
            if words not in stopwords.words("english"):
                new_text.append(words)
            else:
                new_text.append("")
        text = " ".join(new_text)

        return text


pipe = Pipeline(
    [
        ("lower", LowerCasing()),
        ("remove punctuation", RemovePunctuation()),
        ("remove accent", RemoveAccent()),
        ("remove stopwords", RemoveStopWords()),
    ]
)


# ---------------------------------------------------------------------------
# ➡️➡️➡️➡️➡️ GLOVE IMPORT
# ----------------------------------------------------------------
# ----------------------------------------------------------------


class Glove:
    """
    The class is responsible for importing the saved gensim word2vec 200 dim vector and use it to encode the question

    """

    def __init__(self, path):
        self.wv = KeyedVectors.load(path)

    def sent_vec(self, sent):
        """
        Creates a vector from sentence
        """
        vector_size = self.wv.vector_size
        wv_res = np.zeros(vector_size)
        ctr = 1
        for w in sent:
            if w in self.wv:
                ctr += 1
                wv_res += self.wv[w]
        wv_res = wv_res / ctr
        return wv_res


word_2_vec_model_dir = model_dir_ + "glove_model"


# ---------------------------------------------------------------------------
# ➡️➡️➡️➡️➡️ DATA LOADER
# ----------------------------------------------------------------
# ----------------------------------------------------------------


class LoadData:
    def __init__(self, embedding_path="", context_path=""):
        self.embedding_path = embedding_path
        self.context_path = context_path

    def load_context_embeddings(self):
        # This will load premade embeddings
        with open(self.embedding_path, "rb") as file:
            vec = pickle.load(file)

        # Convert into dataframe
        vec = pd.DataFrame(vec)

        return vec

    def load_context_data(self):
        # This will load the original data for context
        og_context = pd.read_csv(self.context_path)
        return og_context


embedding_path = model_dir_ + "glove_encoding.pkl"
context_path = data_dir + "context_data.csv"


# ---------------------------------------------------------------------------
# ➡️➡️➡️➡️➡️ FIND COSINE SIMILARITY
# ----------------------------------------------------------------
# ----------------------------------------------------------------


def find_cosine_similarity(vec, question_vector):
    """
    Takes the embedding dataframe of the context and the embedding of the question
    Finds the cosine similarity between two
    Gives the index whose cosine similarity is maximum
    """
    vec["cosine_similarity"] = vec["context"].apply(
        lambda x: 1 - cosine(x, question_vector)
    )  # Applies the cosine similarity and store in a new column
    index_max_similarity = vec[
        "cosine_similarity"
    ].argmax()  # Finds the index with maximum cosine similarity

    return index_max_similarity


# --------------------------------------------------------------------
# ➡️➡️➡️➡️➡️ FORMAT DATA
# ----------------------------------------------------------------


def format_data(context_sentence, question):
    to_predict = [
        {
            "context": context_sentence,
            "qas": [
                {
                    "question": question,
                    "id": "0",
                }
            ],
        }
    ]

    return to_predict


# -----------------------------------------------------------------------
# ➡️➡️➡️➡️➡️ MODEL
# ----------------------------------------------------------------
# ----------------------------------------------------------------


class Model:
    def __init__(self):
        self.model = QuestionAnsweringModel(
            "bert", model + "/best_model/", use_cuda=False
        )

    def predict(self, to_predict):
        predictions, raw_input = self.model.predict(to_predict)
        return predictions[0]["answer"][0]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#                                           FLASK APP
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

app = Flask(__name__)


# This is the home page of our site


# When we click to the start chat button it should lead
# to the chatbot page
def get_sample_query():
    """
    This function returns randomly sampled queries 3 to be specific and pass it to the html

    Returns:
    query_to_suggest (List) - list of strings
    """
    sample_query = [
        "What is experience required for WLAN Device Driver Development Engineer - Linux post in Strivex Consulting Pvt Ltd company?",
        "What is experience required for Senior Developer (retail POS Development) post in Careernet Technologies Pvt Ltd hiring for Senior Developer (retail POS Development) company?",
        "What is role for WLAN Device Driver Development Engineer - Linux post in Strivex Consulting Pvt Ltd company?",
        "What is role for Senior Manager,deputy General Manager- Digital Marketing post in HumanKonnect company?",
        "What is role for Big Data Test Engineer post in Bob Technologies company?",
        "What are skills required for Big Data Hadoop Developer post in dynproindia company?",
        "What are skills required for Marketing Executive-chennai post in Apollo Sugar Clinics Limited company?",
        "What are skills required for Oracle Corporate Trainer post in Koenig Solutions Ltd company?",
        "What is experience required for Sales Manager - Bangalore post in O&G SKILLS INDIA PVT LTD company?",
        "What is experience required for Sr Technical Lead with a Product Based Company post in Confidential company?",
        "What is experience required for Bench Sales Recruiter post in Confidential company?",
        "What is experience required for HR Operation Lead post in Confidential company?",
        "What is experience required for Sr FPGA Engineer - Bangalore post in Gobrah Management Consulting Services Pvt Ltd hiring for a reputed Semiconductor company (800+ Employees) company?",
        "What are skills required for Currently we have an Immediate Requirement for SAP SD Consultant post in Marlabs Software Pvt Ltd company?",
        "What is salary provided by Covalense Technologies Private Limited company for Java - SSE , Technical Lead post?",
        "What is salary provided by Bharath Infra Exports and Imports Ltd company for Sales Executive for Tiles post?",
    ]

    query_to_suggest = random.sample(sample_query, 3)
    return query_to_suggest


@app.route("/")
def chatbot():
    query_to_suggest = (
        get_sample_query()
    )  # Calls the query function to get randomly sampled strings
    return render_template("index.html", sample=query_to_suggest)


# you can get input using 2 methods GET and POST(Input without URL)
@app.route("/predict", methods=["POST"])
def predict():
    # This will accept the question written in the chatbot
    message = request.form["message"]

    processed_question = pipe.fit_transform(
        message)  # Preprocesses the question
    question_vector = Glove(word_2_vec_model_dir).sent_vec(
        processed_question
    )  # Create embedding of question
    ld = LoadData(
        embedding_path=embedding_path, context_path=context_path
    )  # Loading the pre-made context embeddings and context data
    vec = ld.load_context_embeddings()
    context_data = ld.load_context_data()
    index_max_similarity = find_cosine_similarity(
        vec, question_vector
    )  # Find the index of maximum cosine similarity

    # Fetching the context from the original context data
    context = context_data.iloc[
        index_max_similarity, 0
    ]  # This will have context of the question stored to it

    to_predict = format_data(context, message)
    # Formatting the original question and context.
    # Not using the processed question because the model is not trained on
    # processed questions

    answer = Model().predict(to_predict)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
