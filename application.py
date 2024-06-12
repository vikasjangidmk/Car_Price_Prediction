from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the model and dataset with error handling
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'LinearRegressionModel.pkl' is in the correct location.")

try:
    car = pd.read_csv('Cleaned_Car_data.csv')
except FileNotFoundError:
    raise Exception("CSV file not found. Ensure 'Cleaned_Car_data.csv' is in the correct location.")

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    try:
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
        result = str(np.round(prediction[0], 2))
    except Exception as e:
        result = f"Error in prediction: {str(e)}"

    return result

if __name__ == '__main__':
    app.run(debug=True)
