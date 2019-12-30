'''
Models.py: All functions needed to model our DNN with goal to find Recession
Periods on financial markets
'''

'''
MIT License

Copyright (c) 2017 Simao Luiz Stanislawski Junior (simaoluizjr@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
All Imports (are explained in README.txt)
'''
import h2o
import pandas as pd
import numpy as np
from   sklearn.model_selection import train_test_split



def TrainPreliminarModel(dataset):
    ''' Uses the dataset with the preliminar data (described in pdf report), trains
        a DNN and returns the H2O Model. It returns the testframe and the model
        returns testframe, h2o_model   (the testframe returned is need so we can validate results)
    '''
    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    
    h2o.init()
    #h2o.connect()
    
    #Divides dataset proportion 80/20
    X_train, X_test, y_train, y_test = train_test_split(dataset.ix[:,:-1], dataset.ix[:,9], 
                                                        test_size=0.20, random_state=None)

    train = X_train
    train["PotentialCrisis"] = y_train
    test = X_test
    test["PotentialCrisis"] = y_test

    #Converts pandas to H2OFrame
    train = h2o.H2OFrame(python_obj=train)
    test =  h2o.H2OFrame(python_obj=test)

    data_names = train.names[:]
    data_names.remove("PotentialCrisis")

    # Simple Deep Learning Constructor
    data_dl = H2ODeepLearningEstimator(variable_importances=True,
                                      loss ="Automatic",
                                      activation = "rectifier_with_dropout",
                                      balance_classes = True,
                                      hidden = [300,300, 300],
                                       hidden_dropout_ratios = [0.5, 0.5, 0.5],
                                       input_dropout_ratio = 0.0
                                      )

    #Training the Network
    data_dl.train(x                =data_names,
                  y                ="PotentialCrisis",
                  training_frame  = train,
                   validation_frame=test)
    return test, data_dl

'''
Will Train the Final Model and validate using Cross Validation with 5 folds
'''
def TrainCrossValidation5FoldFinalModel(dataset):
    ''' Uses the dataset with the tuned data, trains 5 folds of data with
        a DNN and returns the H2O Model. This model will be Final Model as
        described in pdf report
        returns h2o_model
    '''
    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator    
    h2o.init()
    #h2o.connect()
    
    #Divides dataset proportion 80/20
    X_train, X_test, y_train, y_test = train_test_split(dataset.ix[:,:-1], dataset.ix[:,9], 
                                                        test_size=0.01, random_state=None)
    train = X_train
    train["PotentialCrisis"] = y_train
    test = X_test
    test["PotentialCrisis"] = y_test

    #Converts pandas to H2OFrame
    train = h2o.H2OFrame(python_obj=train)
    test =  h2o.H2OFrame(python_obj=test)

    data_names = train.names[:]
    data_names.remove("PotentialCrisis")
    data_dl = H2ODeepLearningEstimator(variable_importances=True,
                                  loss ="Automatic",
                                  activation = "rectifier_with_dropout",
                                  balance_classes = True,
                                  hidden = [300,300,300],
                                  hidden_dropout_ratios = [0.5,0.5,0.5],
                                  input_dropout_ratio = 0.0,
                                  nfolds = 5,
                                  fold_assignment = 'AUTO'                                   
                                  )


    data_dl.train(x  =data_names,
                  y  ="PotentialCrisis",
                  training_frame  =train,
                  validation_frame=test)
    return data_dl


