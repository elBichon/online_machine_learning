import re
import requests
import pandas as pd
import io
import spacy

url="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

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
	try:
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
	except:
		return False



