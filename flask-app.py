from click import pass_context
from flask import Flask, request, render_template
import joblib
import numpy as np
import os

port = int(os.environ.get('PORT', 5000))

filepath = 'models\Fish_model.pkl'
ml_model = joblib.load(open(filepath,'rb'))

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        
        length_1 = request.form.get('Length_1')
        length_2 = request.form.get('Length_2')
        length_3 = request.form.get('Length_3')
        height = request.form.get('Height')
        width = request.form.get('Width')
        weight = request.form.get('Weight')
        
        try:
            predict = predictionDone(length_1, length_2, length_3, height, width, weight)
            return render_template('predict.html', prediction = predict)
        
        except ValueError:
            return "Please enter valid values"
    pass

def predictionDone(l1, l2, l3, h, wi, we):

    data = [l1, l2, l3, h, wi, we]
    data = [float(n) for n in data]

    data = np.array(data)
    data = data.reshape(1,-1)

    prediction = ml_model.predict(data)

    return prediction

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True, port = port)