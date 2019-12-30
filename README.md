# recessionmeter
﻿***************************************************************************************************
Classificador RecessionMeter
***************************************************************************************************
Projeto Capstone Udacity: Engenheiro de Machine Learning 2017
Por Simao Luiz Stanislawski Junior

---===Conteudo:===---
Examples.py: arquivo com os usos recomendados para este programa
Models.py: arquivo com os modelos de ML necessarios
PrepareData.py: arquivo com as rotinas de download e preprocessamento do dataset
crisis.csv: planilha contendo as datas e a classificaçao dos eventos (PotentialCrisis ou Normal)

---===Instruções de Uso:===---
Use conforme os exemplos mostrados em Examples.py

---===Documentaçao:===---
Os módulos Models.py e PrepareData.py estão documentados em arquivos html com seus respectivos nomes.

---===Dependências:===---
- Java
- Biblioteca de Machine Learning H2O (Para uso do classificador 'DeepLearning')

	Documentaçao: http://docs.h2o.ai/h2o/latest-stable/index.html
	Instalando o modulo H2O python: 

	pip install requests
	pip install tabulate
	pip install scikit-learn

	# The following command removes the H2O module for Python.
	pip uninstall h2o

	# Next, use pip to install this version of the H2O Python module.
	pip install https://h2o-release.s3.amazonaws.com/h2o/rel-tverberg/4/Python/h2o-3.10.3.4-py2.py3-none-any.whl

- Biblioteca da Base de Dados Quandl (Fonte dos dados)

	Documentaçao: https://www.quandl.com/tools/python
	Instalando o api quandl python: pip install quandl

- Outros: sklearn, dateutil, pandas, numpy


---===Problemas com limite de download Quandl: ===---
E um problema improvavel mas se ocorrer basta cadastrar-se gratuitamente no site www.quandl.com e conseguir uma api, e um processo bem rapido.
