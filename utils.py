import re
import requests
import pandas as pd
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# URL-link validation
ip_middle_octet = u"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5]))"
ip_last_octet = u"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"

URL_PATTERN = re.compile(
	u"^"
	# protocol identifier
	u"(?:(?:https?|ftp|rtsp|rtp|mmp)://)"
	# user:pass authentication
	u"(?:\S+(?::\S*)?@)?"
	u"(?:"
	u"(?P<private_ip>"
	# IP address exclusion
	# private & local networks
	u"(?:localhost)|"
	u"(?:(?:10|127)" + ip_middle_octet + u"{2}" + ip_last_octet + u")|"
	u"(?:(?:169\.254|192\.168)" + ip_middle_octet + ip_last_octet + u")|"
	u"(?:172\.(?:1[6-9]|2\d|3[0-1])" + ip_middle_octet + ip_last_octet + u"))"
	u"|"
	# IP address dotted notation octets
	# excludes loopback network 0.0.0.0
	# excludes reserved space >= 224.0.0.0
	# excludes network & broadcast addresses
	# (first & last IP address of each class)
	u"(?P<public_ip>"
	u"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
	u"" + ip_middle_octet + u"{2}"
	u"" + ip_last_octet + u")"
	u"|"
	# host name
	u"(?:(?:[a-z\u00a1-\uffff0-9_-]-?)*[a-z\u00a1-\uffff0-9_-]+)"
	# domain name
	u"(?:\.(?:[a-z\u00a1-\uffff0-9_-]-?)*[a-z\u00a1-\uffff0-9_-]+)*"
	# TLD identifier
	u"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
	u")"
	# port number
	u"(?::\d{2,5})?"
	# resource path
	u"(?:/\S*)?"
	# query string
	u"(?:\?\S*)?"
	u"$",
	re.UNICODE | re.IGNORECASE
                       )
def url_validate(url):
	try:
		if len(url) > 0 and isinstance(url,str) == True:
			url = url.rstrip().lstrip()
			if re.compile(URL_PATTERN).match(url) != None:                                                                                                                                      
				return True
			else:
				return False
		else:
			return False
	except:
		return False


def http_check(url):
	try:
		if len(url) > 0 and isinstance(url,str) == True: 
			r = requests.get(url)
			status_code = r.status_code
			if status_code == 200:
				return True
			else:
				return False
		else:
			return False
	except:
		return False


def read_csv(url):
	try:
		if len(url) > 0 and isinstance(url,str) == True:
			df = pd.read_csv(url)
			return df
		else:
			return False
	except:
		return False


def remove_unique_feature(df):
	try:
		if isinstance(df,pd.DataFrame) == True and len(df.columns) > 1:
			df = df.drop_duplicates()
			i = 0
			features_list = df.columns 
			while i < len(features_list): 
				if len(df[features_list[i]].unique()) == 1: 
					df.drop(features_list[i], 1, inplace=True) 
				else:
					pass
				i += 1 
			return df
		else:
			return False
	except:
		return False


def hasNumbers(inputString):
	try:
		if len(inputString) > 0:
			return any(char.isdigit() for char in inputString)
		else:
			return False	
	except:
		return False

def remove_name(nlp,df):
	columns_to_remove =  []
	for column in df.columns:
		if df[column].dtypes == 'object':
			if hasNumbers(str(df[column].values.tolist()[0]).lower()) == True:
				pass
			else:
				doc = nlp(re.sub("[^a-z]"," ",str(df[column].values.tolist()[0]).lower()))
				for token in doc:
					if token.pos_ == 'PROPN' and token.tag_ == 'NNP' and  token.dep_ == 'compound':
						columns_to_remove.append(column)
	return(list(set(columns_to_remove)))


def data_encoder(df):
	try:
		for col in df.columns: 
			if df[col].dtype == 'object': 
				data = list(set(df[col].values.tolist()))
				dict = {}
				i = 0
				while i < len(data):
					dict[data[i]] = i
					i += 1
				df = df.replace(dict)
			else:
				pass
		return(df)
	except:
		return False


def treat_na(df):
	try:
		i = 0 
		features_list = df.columns 
		while i < len(features_list): 
			df[features_list[i]].fillna(df[features_list[i]].mean()) 
			i += 1 
		return df
	except: 
		return False


def outliers_removal(df):
	try:
		for feature in df.columns: 
			Q1 = np.percentile(df[feature], q=25) 
			Q3 = np.percentile(df[feature], q=75) 
			interquartile_range = Q3-Q1 
			step = 1.5 * interquartile_range 
			outliers = [] 
			df = df.drop(df.index[outliers]).reset_index(drop = True)    
			return df
	except:
		return False


def remove_colinar_features(label,features,df):
	try:
		corr_matrix = df[features].corr(method='spearman').abs()
		upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
		to_drop = [column for column in upper.columns if any(upper[column] < 0.5)]
		df = df.drop(df[to_drop], axis=1)
		return df
	except:
		return False


def remove_index(df):
	try:
		i = 0
		for col in df.columns:
			if len(list(set(df[col].values.tolist()))) == len(df):
				df.drop(col, inplace=True, axis=1)
			else:
				pass
		return df
	except:
		return False


def scale_data(df):
	try:
		scaler = StandardScaler()
		StandardScaler(copy=True, with_mean=True, with_std=True)
		scaler.fit(df)
		df = scaler.transform(df)
		return df
	except:
		return False


def create_train_test(df,label):
	X_train, X_test, y_train, y_test = train_test_split(df, df.label,train_size=0.9, test_size=0.1)
	return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}



def confusion_matrix(y_true,y_pred):
	try:
		return confusion_matrix(y_true, y_pred)
	except:
		return False


def get_accuracy_score(y_true,y_pred):
	try:
		return  accuracy_score(y_true, y_pred)
	except:
		return False

def get_mean_square_error(y_true,y_pred):
	try:
		return mean_squared_error(y_true, y_pred)
	except:
		return False

def create_model(names,classifiers,x_train,y_train,x_test,y_test):
	score_list = []
	for name, clf in zip(names, classifiers):
		clf.fit(x_train,y_train)
		score = clf.score(x_test,y_test)
		score_list.append(score)
	return score_list


