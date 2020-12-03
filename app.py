# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:17:31 2020

@author: shahe
"""

#Import necessary libraries
from flask import Flask, render_template,request, url_for
import pandas as pd
import numpy as np
import pickle

#load the saved model
saved_model = pickle.load(open('rf_reg_model.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])

def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])

def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Kms_Driven_log = np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type_Petrol = request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol == 'Petrol'):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        elif(Fuel_Type_Diesel == 'Diesel'):
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
        Year = int(request.form['Year'])
        Year = 2020 - Year
        
        Seller_Type_Individual = request.form['Seller_Type_Individual']
        if(Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0
        Transmission_Manual = request.form['Transmission_Manual']
        if(Transmission_Manual == 'Manual'):
            Transmission_Manual = 1
        else:
            Transmission_Manual = 0
        prediction = saved_model.predict([[Present_Price,Kms_Driven_log,Year,Owner,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]])
        output = round(prediction[0],2)
        
        if output < 0:
            return render_template('index.html',prediction_texts = "Sorry,selling price for the choosen car is not available")
        else:
            return render_template('index.html',prediction_texts = "This car can be sold at ₹{} lakhs".format(output))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
    



















