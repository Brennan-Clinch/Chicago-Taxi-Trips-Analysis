#!/usr/bin/env python
# coding: utf-8

# In[12]:


from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
app = Flask(__name__,template_folder='templates')
model = pickle.load(open("C://Users//JCCLI//Downloads//LightGBM_Chicago_Taxi.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('TaxiFarePrediction.html')
@app.route('/predict',methods =['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('TaxiFarePrediction.html', prediction_text = 'Your Fare will be close to $ {}'.format(output))

if __name__ == '__main__':
    app.run(port=5000)

