from flask import Flask, request, jsonify
from flask_cors import CORS
from main import clean_input_data, predictValue

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todos os endpoints

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = clean_input_data(data['input'])
    
    prediction = predictValue(X)
    print(prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)