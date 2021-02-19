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
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

app = Flask(__name__, static_folder="static")


@app.route("/") 
def home():
	return render_template("index.html")

@app.route("/result", methods=["POST", "GET"]) 
def result_page():
	try:
		if request.method == "POST":
			url = str(request.form["url"]).rstrip().lstrip()
			if len(url) > 0 and isinstance(url, str) == True and len(str(request.form["target"])) > 0 and isinstance(request.form["target"], str) == True and request.form["type"] in ['regression','classification', 'nlp_en'] and utils.url_validate(url) == True and utils.http_check(url) == True:
				label = str(request.form["target"])
				compute_type = str(request.form["type"])
				df = utils.read_csv(url)
				if label in df.columns:
					if compute_type != 'nlp_en':
						df = utils.data_encoder(df)
						df = utils.remove_unique_feature(df)
						df = utils.remove_index(df)
						df = utils.treat_na(df)
						df = utils.outliers_removal(df)
						target_column = df[label].values.tolist()
						colinear_features = utils.remove_colinar_features(label,df.columns,df)
						df = df[colinear_features]
						df = df.drop(columns=label)
						df['label'] = target_column
						data = utils.create_train_test(df,label)
						normalized_x_train = pd.DataFrame(utils.scale_data(data['X_train']))
						normalized_x_test = pd.DataFrame(utils.scale_data(data['X_test']))
						if compute_type == 'classification':
							names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"]
							classifiers = [KNeighborsClassifier(3),SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),GaussianProcessClassifier(1.0 * RBF(1.0)),DecisionTreeClassifier(max_depth=5),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),MLPClassifier(alpha=1, max_iter=1000),AdaBoostClassifier(),GaussianNB(),QuadraticDiscriminantAnalysis()]
							score_list = utils.create_model(names,classifiers,normalized_x_train, data['y_train'],normalized_x_test, data['y_test'])
							score_index = score_list.index(max(score_list))
							result = 'the best features are: ' + str(df.columns) + ' the best model is ' +str(classifiers[score_index])
						elif compute_type == 'regression':
							names = ["Nearest Neighbors", "Linear RBF", "Linear SVM", "Linear Polynomial", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]
							regressors = [KNeighborsRegressor(n_neighbors=3),SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),SVR(kernel='linear', C=100, gamma='auto'),SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1),DecisionTreeClassifier(max_depth=5),RandomForestRegressor(max_depth=2, random_state=0),MLPRegressor(random_state=1, max_iter=500),AdaBoostRegressor(random_state=0, n_estimators=100)]
							score_list = utils.create_model(names,regressors,normalized_x_train, data['y_train'],normalized_x_test, data['y_test'])
							score_index = score_list.index(max(score_list))
							result = 'the best features are: ' + str(df.columns) + ' the best model is ' +str(regressors[score_index])
					elif compute_type == 'nlp_en':
						en_core = spacy.load('en_core_web_sm')
						df = utils.encode_target_nlp(df, label)
						text_field = str(df.columns.drop(label).values[0])
						df = utils.standardize_text(df, text_field)
						stop = spacy.lang.en.stop_words.STOP_WORDS
						df[text_field] = df[text_field].str.split().apply(lambda x: [item for item in x if item not in stop])
						df[text_field] = df[text_field].apply(lambda x: [item for item in x if item not in stop])
						df[text_field] = [','.join(map(str, l)) for l in df[text_field]]
						df[text_field] = df[text_field].str.replace(',',' ')
						df[text_field] = df[text_field].apply(lambda x: " ".join([y.lemma_ for y in en_core(x)]))
						vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, stop_words = None, max_features = 50) 
						train_data_features = vectorizer.fit_transform(df[text_field].values.tolist()[0:5])
						train_data_features = train_data_features.toarray()[0:5]
						forest = RandomForestClassifier(random_state=42)
						param_grid = {'n_estimators': [5],'max_features': ['auto'],'max_depth' : [5],'criterion' :['gini']}
						forest = GridSearchCV(estimator=forest, param_grid=param_grid)
						forest = forest.fit(train_data_features, df[label][0:5])
						params = forest.best_params_
						result = 'Due to limitation of heroku, grid search is done with single parameters and 1 fold cross-validation (I know this does not make sense) the best parameters are: ' + str(params)
						#result = 'Due to the limitations of heroku, for now it is static and defaut implementation of random forest by sklearn'
					return render_template("result.html", result = result)
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
