from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("/home/pankaj/Documents/code/TASK2GRIP/data/StudyHour")
    return render_template("preview.html",df_view = df)

@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		studyhours = request.form['studyhours']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [studyhours]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'Regression':
		    logit_model = joblib.load('/home/pankaj/Documents/code/TASK2GRIP/data/model.pkl')
		    result_prediction = logit_model.predict(ex1)

	return render_template('index.html', studyhours = studyhours,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)