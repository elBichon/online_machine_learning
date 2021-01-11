import utils
import urllib
import pandas as pd
import spacy

def test_url_validate():
	assert utils.url_validate('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv') == True
def test_url_validate1():
	assert utils.url_validate('dd') == False
def test_url_validate2():
	assert utils.url_validate('') == False
def test_url_validate3():
	assert utils.url_validate(3) == False

def test_http_check1():
	assert utils.http_check('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv') == True
def test_http_check2():
	assert utils.http_check('https://raw.githubusercontent.com/azertyuiop') == False
def test_http_check3():
	assert utils.http_check('') == False
def test_http_check4():
	assert utils.http_check('123') == False
def test_http_check5():
	assert utils.http_check(3) == False

def test_read_csv1():
	assert isinstance(utils.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv'),pd.DataFrame) == True
def test_http_check2():
	assert isinstance(utils.read_csv(''),pd.DataFrame)== False
def test_http_check3():
	assert isinstance(utils.read_csv('123'),pd.DataFrame) == False
def test_http_check4():
	assert isinstance(utils.read_csv(3),pd.DataFrame) == False


def test_remove_unique_feature1():
	df = utils.read_csv('https://raw.githubusercontent.com/elBichon/online_machine_learning/main/titanic/train.csv')
	assert isinstance(utils.remove_unique_feature(df),pd.DataFrame) == True
def test_remove_unique_feature2():
	assert isinstance(utils.remove_unique_feature(''),pd.DataFrame) == False
def test_remove_unique_feature3():
	assert isinstance(utils.remove_unique_feature('12132'),pd.DataFrame) == False
def test_remove_unique_feature3():
	data = {'col_1': [3, 2, 1, 0]}
	df = pd.DataFrame.from_dict(data)
	assert isinstance(utils.remove_unique_feature(3),pd.DataFrame) == False

