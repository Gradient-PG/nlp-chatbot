from flask import Flask, render_template, request, jsonify
import requests

def get_response(text):
    return requests.post("http://127.0.0.1:8080/wfpredict/wf", text).text


app = Flask(__name__)

@app.route("/", methods =["GET"])
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    #todo validation
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)
if __name__ == "__main__":
    app.run(debug=True)

