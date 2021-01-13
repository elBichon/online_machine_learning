import flask
from flask import Flask, render_template, request, flash, session
import utils
import re
import requests
import pandas as pd
import io
import spacy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from tpot import TPOTClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

app = Flask(__name__, static_folder="static")

@app.route("/") 
def home():
	return render_template("index.html")

@app.route("/result", methods=["POST", "GET"]) 
def result_page():
	try:
		if request.method == "POST":
			if len(str(request.form["url"])) > 0 and isinstance(request.form["url"], str) == True and len(str(request.form["target"])) > 0 and  isinstance(request.form["target"], str) == True and request.form["type"] in ['regression','classification'] == True:
				print('toto')
				return render_template("result.html")
		else:
			print('tutu')
			return render_template("error.html")
	except:
			return render_template("error.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
    
if __name__ == "__main__":
	app.debug = True
	app.run()
