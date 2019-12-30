# recessionmeter

## RecessionMeter Classificator

Capstone Project Udacity: Machine Learning Engineer 2017
Completed by Simao Luiz Stanislawski Junior

### Content
Examples.py: file with recommended uses for this software
Models.py: file with necessary Machine Learning Models
PrepareData.py: file with download routines and dataset preprocessing
crisis.csv: spreadsheet with historical dates and historical classification of events (PotentialCrisis* ou Normal)

*PotentialCrisis is correspondent to 'recession'

### Use Instructions
Just follow the examples shown in Examples.py and use your creativity

### Documentation
Models.py and PrepareData.py are documented in html files with their respective names.

### Dependencies
- Java
- Machine Learning H2O Library (For 'DeepLearning' Classificator Use)

	Documentation: http://docs.h2o.ai/h2o/latest-stable/index.html
	Installing H2O python module: 

	pip install requests
	pip install tabulate
	pip install scikit-learn

	# The following command removes the H2O module for Python.
	pip uninstall h2o

	# Next, use pip to install this version of the H2O Python module.
	pip install https://h2o-release.s3.amazonaws.com/h2o/rel-tverberg/4/Python/h2o-3.10.3.4-py2.py3-none-any.whl

- Quandl Database Library (Datasource)

	Documentation: https://www.quandl.com/tools/python
	Installing api quandl python: pip install quandl

- Others: sklearn, dateutil, pandas, numpy
- Please note you always can just use: pip install -r requirements.txt (must have Java installed)

### Known Quandl Download Limit Problem
Although improbable you can solve it signing up free on www.quandl.com and getting an API KEY very quickly.

