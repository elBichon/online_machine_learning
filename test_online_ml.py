import utils
import urllib

def test_check_http():
	assert utils.url_validate('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv') == True
def test_check_http1():
	assert utils.url_validate('dd') == False
def test_check_http2():
	assert utils.url_validate('') == False
def test_check_http3():
	assert utils.url_validate(3) == False

