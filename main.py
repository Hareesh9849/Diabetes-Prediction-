import numpy
import os
from PIL import Image
import numpy as np      # Importing the libraries
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn import metrics
from flask import Flask, render_template, request
import cv2
from IPython.display import Image
import pandas as pd
import os
from flask import Flask

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
    pregnencies=request.form['pregnencies']
    glucose=request.form['glucose']
    blood_pressure=request.form['blood_pressure']
    skin_thickness=request.form['skin_thickness']
    insulin=request.form['insulin']
    bmi=request.form['bmi']
    dpf=request.form['dpf']
    age=request.form['age']
    import numpy as np
    dp1 = pd.read_csv('diab.csv')
    X=dp1.iloc[:1,:-1].values
    n=[int(pregnencies),float(glucose),float(blood_pressure),float(skin_thickness),float(insulin),float(bmi),float(dpf),int(age)]
    arr=numpy.array(n)
    df = pd.DataFrame(columns = [1,2,3,4,5,6,7,8,9])
    data_to_append = {}
    for i in range(len(df.columns)-1):
        data_to_append[df.columns[i]] = arr[i]
    data_to_append[df.columns[i+1]]=1
    df = df.append(data_to_append, ignore_index = True)
    df.to_csv('data.csv')
    c=pd.read_csv('data.csv')
    X=c.iloc[:1,:-1].values
    fin=model.predict(X)
    print(X)
    s=" "
    if(fin==1):
        s="You have Diabetes,Please Consult the Doctor "
    else:
        s="You don't have Diabetes"
    k=prediction_text='predicted = "{}"'.format(s)
    print(k)
    return render_template("upload.html",prediction_text='{}'.format(s))
    

if __name__ == "__main__":
    app.run(port=5000, debug=True)
