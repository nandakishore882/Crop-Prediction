from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
import pickle
from urllib.parse import urlencode
 
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
loaded_model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
loaded_scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), 'rb'))

@app.route('/')
def home():
    nitrogen    = request.args.get('nitrogen', '')
    phosphorus  = request.args.get('phosphorus', '')
    potassium   = request.args.get('potassium', '')
    temperature = request.args.get('temperature', '')
    humidity    = request.args.get('humidity', '')
    ph          = request.args.get('ph', '')
    rainfall    = request.args.get('rainfall', '')
    prediction  = request.args.get('prediction', '')
 
    return render_template('index.html',
        nitrogen    = nitrogen,
        phosphorus  = phosphorus,
        potassium   = potassium,
        temperature = temperature,
        humidity    = humidity,
        ph          = ph,
        rainfall    = rainfall,
        prediction  = prediction,
    )

@app.route('/predict', methods=['POST'])
def predict():
    N        = int(request.form['Nitrogen'])
    P        = int(request.form['Phosporus'])
    K        = int(request.form['Potassium'])
    temp     = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph       = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])
 
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred  = np.array(feature_list).reshape(1, -1)
    input_scaled = loaded_scaler.transform([[N, P, K, temp, humidity, ph, rainfall]])
    prediction   = loaded_model.predict(input_scaled)
 
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
 
    if prediction[0] in crop_dict:
        crop   = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop with the provided data."
 
    params = urlencode({
        'nitrogen':    N,
        'phosphorus':  P,
        'potassium':   K,
        'temperature': temp,
        'humidity':    humidity,
        'ph':          ph,
        'rainfall':    rainfall,
        'prediction':  result,
    })
    return redirect(url_for('home') + '?' + params)

#this is main
if __name__ == '__main__':
    app.run(debug=True)
