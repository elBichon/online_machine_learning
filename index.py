import flask
from flask import Flask, render_template, request, flash, session
import utils
import re
import requests
import pandas as pd
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from tpot import TPOTRegressor
#from tpot import TPOTClassifier
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
			if len(str(request.form["url"])) > 0 and isinstance(request.form["url"], str) == True and len(str(request.form["target"])) > 0 and isinstance(request.form["target"], str) == True and request.form["type"] in ['regression','classification'] and utils.url_validate(request.form["url"]) == True and utils.http_check(request.form["url"]) == True:
				url = request.form["url"]
				label = request.form["target"]		
				compute_type = request.form["type"]
				df = utils.read_csv(url)
				if label in df.columns:
					df = utils.remove_unique_feature(df)
					df = utils.data_encoder(df)
					df = utils.remove_index(df)
					df = utils.treat_na(df)
					df = utils.outliers_removal(df)
					target_column = df[label].values.tolist()
					df = utils.remove_colinar_features(label,df.columns,df)
					df['label'] = target_column
					#data = utils.create_train_test(df,label)
					#if compute_type == 'classification':
					#	normalized_x_train = pd.DataFrame(utils.scale_data(data['X_train']))
					#	model = utils.build_classifier(normalized_x_train, data['y_train'])
					#	return render_template("result.html", result = str(model))#df.columns.values)
					#else:
					#	normalized_x_train = pd.DataFrame(utils.scale_data(data['X_train']))
					#	model = utils.build_regressor(data['X_train'], data['y_train'])
					return render_template("result.html", result = df.columns.values)
				else:
					return render_template("error.html")
			else:
				return render_template("error.html")
		else:
			return render_template("error.html")
	except:
		return render_template("error.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

    
if __name__ == "__main__":
	app.debug = True
	app.run()
