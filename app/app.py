from flask import Flask , request, jsonify






app = Flask(__name__)
@app.route("/")
def home():
    return "Hello World"

@app.route("/predict", methods = ["POST"])
def predict():
    # Write the things I will recieve through request
    question = request.form.get("question")
    context = request.form.get("context")

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