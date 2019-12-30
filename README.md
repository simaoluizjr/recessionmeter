# recessionmeter

## RecessionMeter Classificator

Capstone Project Udacity: Machine Learning Engineer 2017<br>
Completed by Simao Luiz Stanislawski Junior

### Content
Examples.py: file with recommended uses for this software<br>
Models.py: file with necessary Machine Learning Models<br>
PrepareData.py: file with download routines and dataset preprocessing<br>
crisis.csv: spreadsheet with historical dates and historical classification of events (PotentialCrisis* ou Normal)<br>

*PotentialCrisis is correspondent to 'recession'

### Use Instructions
Just follow the examples shown in Examples.py and use your creativity

### Documentation
Models.py and PrepareData.py are documented in html files with their respective names.

### Dependencies
- Java
- Machine Learning H2O Library (For 'DeepLearning' Classificator Use)

	Documentation: http://docs.h2o.ai/h2o/latest-stable/index.html<br>
	Installing H2O python module: <br>

	pip install requests<br>
	pip install tabulate<br>
	pip install scikit-learn<br>
	pip install h2o<br>

	
- Quandl Database Library (Datasource)

	Documentation: https://www.quandl.com/tools/python<br>
	Installing api quandl python: pip install quandl

- Others: sklearn, dateutil, pandas, numpy
- Please note you always can just use: pip install -r requirements.txt (must have Java installed)

### Known Quandl Download Limit Problem
Although improbable you can solve it signing up free on www.quandl.com and getting an API KEY very quickly.

### On Use
Please note deep learning algorithms have some 'margin of error' on classification. I recommend strongly to repeat the training and inference process 10 times and take the average for current period to have an accurate estimate.

### Responsability
Of course, this software use means NO RECOMENDATION of actions and success of its prediction IS NOT GUARANTEED. Use at your own discretion.

