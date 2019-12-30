'''
PrepareData: The data downloading and engineering routines to create a good
             recession classifier n months ahead
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
import quandl
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from dateutil import parser

#Please uncomment this line and put some key if some unprobable problem with quandl
#quandl.ApiConfig.api_key = "your_key_here"

def get_macroeconomic_data():
    '''
    Macroeconomic Historical Indicators
    '''
    business_cycles = 12 #years
    #Alguns dados e necessario fazer um lag para ficarmos em dezembro, desde que seja razoavel

    #Get Industrial Production
    US_IND_PRODUCTION =  quandl.get('FRED/INDPRO')
    #Dont forget to percent rank everything
    US_IND_PRODUCTION = US_IND_PRODUCTION.resample('M').mean()
    indprod_yavg = US_IND_PRODUCTION.rolling(business_cycles * 12).mean()
    US_IND_PRODUCTION = US_IND_PRODUCTION/indprod_yavg
    US_IND_PRODUCTION.columns = ["IndProd"]

    #Get Unemployment
    US_UNEMPLOYMENT = quandl.get("FRED/UNRATE")
    #Dont forget to percent rank everything
    US_UNEMPLOYMENT = US_UNEMPLOYMENT.resample('M').mean()
    unemployment_10yavg = US_UNEMPLOYMENT.rolling(business_cycles * 12).mean()
    US_UNEMPLOYMENT = US_UNEMPLOYMENT/unemployment_10yavg
    US_UNEMPLOYMENT.columns = ["Unemployment"]

    #Get Real Disposable Personal Income
    US_REAL_DISPOSABLE_PERSONAL_INCOME = quandl.get("FRED/DSPIC96")
    #Dont forget to percent rank everything
    US_REAL_DISPOSABLE_PERSONAL_INCOME = US_REAL_DISPOSABLE_PERSONAL_INCOME.resample('M').mean()
    persincome_max = US_REAL_DISPOSABLE_PERSONAL_INCOME.rolling(business_cycles * 12).mean()
    US_REAL_DISPOSABLE_PERSONAL_INCOME = US_REAL_DISPOSABLE_PERSONAL_INCOME/persincome_max
    US_REAL_DISPOSABLE_PERSONAL_INCOME = US_REAL_DISPOSABLE_PERSONAL_INCOME.shift(1)
    US_REAL_DISPOSABLE_PERSONAL_INCOME.columns = ["DispPersIncome"]

    #Get Real Manufacturing Trade Sales since 1967
    US_REAL_MFG_TRADE_SALES = quandl.get("FRED/CMRMTSPL")
    #Dont forget to percent rank everything(TO DO!!!)
    US_REAL_MFG_TRADE_SALES = US_REAL_MFG_TRADE_SALES.resample('M').mean()
    tradesales_max = US_REAL_MFG_TRADE_SALES.rolling(business_cycles * 12).mean()
    US_REAL_MFG_TRADE_SALES = US_REAL_MFG_TRADE_SALES/tradesales_max
    US_REAL_MFG_TRADE_SALES.columns = ["Mfg_Trade_Sales"]

    #US_UNIT_LABOR_COSTS_Q = quandl.get("FRED/ULCBS").resample('M').mean().interpolate()  

    US_REAL_TRADE_W_DOLLAR_IDX = quandl.get("FRED/TWEXBPA").resample('M').mean()
    US_REAL_TRADE_W_DOLLAR_IDX.columns = ["TradeWeightedDollarIdx"]

    #US_SVENY = quandl.get("FED/SVENY", collapse = "monthly").resample('M').mean()

    #Spread Juros de Longo e Curto Prazo: "Crisis are common when this comes to zero"
    US_LT_RATE = quandl.get("FRED/DGS10", collapse = "monthly")
    US_ST_TREASURY_BILL = quandl.get("FRED/DTB3", collapse = "monthly")
    US_DIFF_INTEREST_RATE = US_LT_RATE - US_ST_TREASURY_BILL
    US_DIFF_INTEREST_RATE.columns = ["InterestRateSpread"]
    
    #Will merge every features now
    X_macro =  pd.concat([US_IND_PRODUCTION, US_UNEMPLOYMENT, US_REAL_DISPOSABLE_PERSONAL_INCOME,
                US_REAL_TRADE_W_DOLLAR_IDX, US_DIFF_INTEREST_RATE ], axis=1)
    
    return X_macro

def get_fundamental_data(business_cycles_years):
    '''
    Fundamental Indicators Directly Related to SP500 Companies
    '''

    #GET Robert Shiller's Yale Set for historical SP500 
    ROBERT_SHILLER_SPCOMP_DATA = quandl.get("YALE/SPCOMP", collapse = "monthly") 

    #These are relevant data of Robert Shiller's Yale set: CAPE is a very good stationary indicator of cheapness
    #ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"]
    #ROBERT_SHILLER_SPCOMP_DATA["Cyclically Adjusted PE Ratio"]
    ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"] = ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"].resample('M').mean()
    div_mean = ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"].rolling(business_cycles_years * 12).mean()
    ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"] = ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"]/div_mean

    ROBERT_SHILLER_SPCOMP_DATA["CPI"] = ROBERT_SHILLER_SPCOMP_DATA["CPI"].pct_change().rolling(12).sum()
    
    #Will merge every features now
    X_fundamentals =  pd.concat([ROBERT_SHILLER_SPCOMP_DATA["Real Dividend"],
                                 ROBERT_SHILLER_SPCOMP_DATA["Cyclically Adjusted PE Ratio"],
                                 ROBERT_SHILLER_SPCOMP_DATA["CPI"]], axis=1)
    
    return  ROBERT_SHILLER_SPCOMP_DATA["Real Price"], X_fundamentals

def get_technical_data(price_arr):
    '''
    Technical and Sentiment Indicators
    '''

    #Technical Indicator: Number of Stocks with Prices Advancing or Declining in percentage of all
    adv = quandl.get("URC/NYSE_ADV", collapse = "monthly" )
    dec = quandl.get("URC/NYSE_DEC", collapse = "monthly" )
    US_NYSE_ADVANCES = adv / (adv + dec)
    US_NYSE_ADVANCES.columns = ["NyseAdvances"]

    #Sentiment Meters Indicators: What consumers and managers think of future?
    US_MAN_PMI = quandl.get("ISM/MAN_PMI").resample('M').mean()
    US_MAN_PMI.columns = ["ManufacturingPMI"]

    US_MICHIGAN_CONSUMER_SENTIMENT = quandl.get("UMICH/SOC1").resample('M').interpolate()
    US_MICHIGAN_CONSUMER_SENTIMENT.columns = ["ConsumerSentiment"]


    #Techinical Indicator Relative Strength of 52 week trading
    #max_price = panda_runmax(price_arr, 12).resample('M').mean()
    #min_price = panda_runmin(price_arr, 12).resample('M').mean()
    max_price = price_arr.rolling(12).max()
    min_price = price_arr.rolling(12).min()
    US_W52_POS =(price_arr - min_price)/(max_price - min_price)
    US_W52_POS.name = "W52RelStr"
    
    #Will merge every features now
    X_tech =  pd.concat([US_NYSE_ADVANCES, US_MAN_PMI, US_MICHIGAN_CONSUMER_SENTIMENT, US_W52_POS], axis=1)
    
    return X_tech

def readCrisisCsv():
    #Reads crisis.csv file and transforms into panda
    x = pd.read_csv("crisis.csv")
    return x.ix[:,:-1]

def getCrisisIndex(d, crisis_df):
    #Aux. method to prepare label
    #Iterates through start and end dates (from crisis.csv) to find 
    #at which level the date 'd' is
    for i in range(len(crisis_df)):
        if d <= parser.parse(crisis_df.ix[i,'End']):
            return i   
    return None

def between(dx, d1, d2):
    #Aux. method to prepare_label
    #Checks if date dx is between d1 and d2
    return True if dx >= d1 and dx <= d2 else False

def get_bigfive_preprocessing(X):
    '''
    Transform the main macroeconomic features and consumesentiment in just one variable using PCA
    '''
    bigfive_names = ["IndProd", "Unemployment", "DispPersIncome", "ConsumerSentiment"]
    X_BigFive = X[bigfive_names]
    pca = PCA(n_components=1)
    pca.fit(X_BigFive)

    #print(pca.explained_variance_ratio_) 
    X_BigFive = pca.transform(X_BigFive)
    X_BigFive = pd.DataFrame(X_BigFive, index = X.index, columns=["BigFive"])

    X = X.drop(bigfive_names, axis=1)
    X["BigFive"] = X_BigFive
    return(X)

def get_sentiment_data_preprocessing(X):
    #Transforms Relative Strength for 52 Weeks to Binary Data
    #If Rel Str higher than 90% or lower than 10% is True else is False
    w52 = X["W52RelStr"]
    w52 = ((w52 > 0.9) | (w52 < 0.1)).astype(int)

    #Transforms Nyse Advances Ratio to Binary Data
    #If Advances higher than 85% or lower than 15% is True else is False
    nadv = X["NyseAdvances"]
    nadv = ((nadv>0.85) | (nadv < 0.15)).astype(int)

    X["W52RelStr"] = w52
    X["NyseAdvances"] = nadv
    return(X)


def PrepareLabel(months_ahead, dates_index):
    ''' Prepare Label: Is going to read 'crisis.csv' file and transform into a LABEL "PotentialCrisis" or "Normal"
        months_ahead:  Horizon in months that will be predicted after
        dates_index:   A dates_index from features dataframe, so this method will know what data to search
        returns y (label set)
    '''
    print ("Preparing labels: are you going to train?")
    x = readCrisisCsv()
    lst = list()
    for dx in dates_index:
        i = getCrisisIndex(dx, x)
        current = int(0)
        if i != None:
            current = int(between(dx, parser.parse(x.ix[i,"Start"]), parser.parse(x.ix[i,"End"])))
        lst.append("PotentialCrisis" if current else "Normal")

    df = pd.Series(lst)
    df.index = dates_index
    df.name = "PotentialCrisis"
    #Will shift months_ahead backwards as we want to predict n months ahead crisis
    return df.shift(-months_ahead)


def DownloadXData():
    '''
    Downloads all Quandl Data needed (it must be preprocessed after)
    returns X (features)
    '''
    print ("Downloading data. Please wait a while...")
    X_macro = get_macroeconomic_data()
    price_arr, X_fundamentals = get_fundamental_data(business_cycles_years=12)
    X_tech = get_technical_data(price_arr=price_arr)

    X = pd.concat([X_macro, X_fundamentals, X_tech], axis=1)
    X = X.dropna()
    return X

'''
Feautures Preprocessing Step II - as documented in pdf
'''
def FeaturesPreprocessingII(X):
    print ("Preprocessing data...")
    X = get_bigfive_preprocessing(X)
    X = get_sentiment_data_preprocessing(X)
    return(X)


def PrepareFeatures():
    '''
    Prepare Features: Will Download(DownloadXData), Preprocess (FeaturesPreprocessingII) and Create X Dataframe
    returns X (ready to train)
    '''
    X = DownloadXData()
    X = FeaturesPreprocessingII(X)
    return X


def MergeDataset(X, y):
    '''
    Combines X and y into an Dataframe (to be used in models)
    returns dataset (ready to train)
    '''
    dataset =  pd.concat([X,y], axis=1)
    dataset = dataset.dropna()
    return dataset




