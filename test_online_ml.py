import utils
import urllib

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

