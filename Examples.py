'''
Examples.py: The suggested uses for PrepareData.py and Models.py
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

import PrepareData
import Models
import quandl
import h2o

#Please uncomment this line and put some key if some unprobable problem with quandl
quandl.ApiConfig.api_key = "rFhqT3ot2z_6AnzpB9nU"

#Get all data needed from Quandl, Preprocess I and II and have features Dataframe ready to train
X = PrepareData.PrepareFeatures()

#Prepares label to predict CRISIS 6 months ahead and gets labels Series
y = PrepareData.PrepareLabel(months_ahead= 6, dates_index= X.index)

#Creates one dataset
dataset = PrepareData.MergeDataset(X,y)

#Trains our preliminar model as documented on pdf
testframe, prel_model = Models.TrainPreliminarModel(dataset)

#Gets Performance Summary on preliminar model
prel_model.model_performance(testframe)

#Trains our final model as documented on pdf with 5 fold CV
final_model = Models.TrainCrossValidation5FoldFinalModel(dataset)

#Please observe the Confusion Matrix that is related to "cross-validation data"
#This model can be persisted for some 12 months, after it's recommended to train again
print (final_model.cross_validation_metrics_summary())

#Now let's predict the last 3 observations, that as of feb/2017 are 10-11-12/2016
#So predictions would be to 04-05-06/2017, if you put 6 months on months_ahead
print ("Final Predictions \n", final_model.predict(h2o.H2OFrame(X.tail(3))))

