import flask
from flask import Flask, render_template, request, flash, session
import utils
import re
import requests
import pandas as pd
import io
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor


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
					data = utils.create_train_test(df,label)
					normalized_x_train = pd.DataFrame(utils.scale_data(data['X_train']))
					normalized_x_test = pd.DataFrame(utils.scale_data(data['X_test']))
					if compute_type == 'classification':
						names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"]
						classifiers = [
						    KNeighborsClassifier(3),
						    SVC(kernel="linear", C=0.025),
						    SVC(gamma=2, C=1),
						    GaussianProcessClassifier(1.0 * RBF(1.0)),
						    DecisionTreeClassifier(max_depth=5),
						    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
						    MLPClassifier(alpha=1, max_iter=1000),
						    AdaBoostClassifier(),
						    GaussianNB(),
						    QuadraticDiscriminantAnalysis()]
						score_list = utils.create_model(names,classifiers,normalized_x_train, data['y_train'],normalized_x_test, data['y_test'])
						score_index = score_list.index(max(score_list))
						return render_template("result.html", result = str(classifiers[score_index]))

					else:
						names = ["Nearest Neighbors", "Linear RBF", "Linear SVM", "Linear Polynomial", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]
						regressors = [
							KNeighborsRegressor(n_neighbors=3),
							SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
							SVR(kernel='linear', C=100, gamma='auto'),
							SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1),
							DecisionTreeClassifier(max_depth=5),
							RandomForestRegressor(max_depth=2, random_state=0),
							MLPRegressor(random_state=1, max_iter=500),
							AdaBoostRegressor(random_state=0, n_estimators=100)]
						score_list = utils.create_model(names,regressors,normalized_x_train, data['y_train'],normalized_x_test, data['y_test'])
						score_index = score_list.index(max(score_list))
						return render_template("result.html", result = str(regressors[score_index]))
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
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)
