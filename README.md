# recessionmeter

## RecessionMeter Classificator

Capstone Project Udacity: Machine Learning Engineer 2017<br>
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
	pip install h2o

	
- Quandl Database Library (Datasource)

	Documentation: https://www.quandl.com/tools/python
	Installing api quandl python: pip install quandl

- Others: sklearn, dateutil, pandas, numpy
- Please note you always can just use: pip install -r requirements.txt (must have Java installed)

### Known Quandl Download Limit Problem
Although improbable you can solve it signing up free on www.quandl.com and getting an API KEY very quickly.

