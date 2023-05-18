import numpy
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn import metrics
from flask import Flask, render_template, request
import cv2
from IPython.display import Image
import pandas as pd

app = Flask(__name__, template_folder=r'C:\Users\haree\DiabetesFinal Project\Final Project\Diabetes Prediction\templates')

model = pickle.load(open('mymodel.pkl','rb'))

@app.route('/')
def index():
    return render_template("image.html")

@app.route('/a',methods=['GET','POST'])
def a():
    return render_template("form.html")

@app.route('/upload',methods=['GET','POST'])
def upload():
    try:
        pregnencies = int(request.form['pregnencies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
    except ValueError:
        return render_template("form.html", error_message="Please enter valid numbers for all fields.")

    dp1 = pd.read_csv('diab.csv')
    X = dp1.iloc[:1,:-1].values
    n = [pregnencies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    arr = numpy.array(n)
    df = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,9])
    data_to_append = {}
    for i in range(len(df.columns)-1):
        data_to_append[df.columns[i]] = arr[i]
    data_to_append[df.columns[i+1]] = 1
    df = df.append(data_to_append, ignore_index=True)
    df.to_csv('data.csv')
    c = pd.read_csv('data.csv')
    X = c.iloc[:1,:-1].values
    fin = model.predict(X)
    if fin == 1:
        s = "You have diabetes. Please consult the doctor."
    else:
        s = "You don't have diabetes."
    k = 'predicted = "{}"'.format(s)
    return render_template("upload.html", prediction_text=s, pregnencies=pregnencies, glucose=glucose, blood_pressure=blood_pressure, skin_thickness=skin_thickness, insulin=insulin, bmi=bmi, dpf=dpf, age=age)

@app.route('/result', methods=['POST'])
@app.route('/result', methods=['POST'])
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Retrieve the entered values from the form
        pregnencies = int(request.form['pregnencies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Perform the prediction using the input values
        input_data = [pregnencies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        prediction = model.predict([input_data])[0]

        # Pass the prediction result and the input values to the template
        return render_template("result.html", 
                               pregnancies=pregnencies, 
                               glucose=glucose, 
                               blood_pressure=blood_pressure, 
                               skin_thickness=skin_thickness, 
                               insulin=insulin, 
                               bmi=bmi, 
                               dpf=dpf, 
                               age=age, 
                               prediction_text="You have diabetes. Please consult the doctor." if prediction==1 else "You don't have diabetes.")



    

if __name__ == "__main__":
    app.run(port=5000, debug=True)
