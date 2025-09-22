from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import sklearn
import requests

print(sklearn.__version__)

# loading models
dtr = pickle.load(open('models/dtr.pkl','rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl','rb'))

# flask app
app = Flask(__name__)

API_KEY = "023297fd0c06ff8ed14cc93597fe4c33"

@app.route('/')
def index():
    lat, lon = 22.5726, 88.3639
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    temperature = response['main']['temp']
    rainfall = response.get('rain', {}).get('1h', 0)

    return render_template('index.html',
                           default_temp=temperature,
                           default_rainfall=rainfall)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # collect inputs
        Crop_Year = request.form['Crop_Year']
        Annual_Rainfall = request.form['Annual_Rainfall']
        Annual_Temperature = request.form['Annual_Temperature']
        Pesticide = request.form['Pesticide']
        Fertilizer = request.form['Fertilizer']
        Area = request.form['Area']
        State = request.form['State']
        Crop = request.form['Crop']
        Season = request.form['Season']

        # Build a DataFrame with the SAME column names used during training
        input_df = pd.DataFrame([{
            "Crop_Year": Crop_Year,
            "Annual_Rainfall": Annual_Rainfall,
            "Annual_Temperature": Annual_Temperature,
            "Pesticide": Pesticide,
            "Fertilizer": Fertilizer,
            "Area": Area,
            "State": State,
            "Crop": Crop,
            "Season": Season
        }])

        # Transform & predict
        transformed_features = preprocessor.transform(input_df)
        prediction = dtr.predict(transformed_features)
        predicted_value = float(prediction[0])

        return render_template(
            'index.html',
            prediction=predicted_value*1000,
            default_temp=Annual_Temperature,
            default_rainfall=Annual_Rainfall
        )

if __name__ == "__main__":
    app.run(debug=True)
