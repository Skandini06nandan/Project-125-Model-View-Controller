from flask import Flask, jsonify, request
from main import getPrediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ["POST"])
def predictAlphabet():
    image = request.files.get("alphabet")
    prediction = getPrediction(image)

    return jsonify({
        "prediction" : prediction
    }),200

if(__name__ == "__main__"):
    app.run(debug = True)