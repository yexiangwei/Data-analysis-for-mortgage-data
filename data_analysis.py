import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.formula.api as smf
import time

#---------------------------start: logit regression class-------------------------------#

class logitRegression():
    def __init__(self, data):
        self.data = data
        self.data = self.data.convert_objects(convert_numeric=True)

        self.var = ['addData', 'zip', 'state', 'propType', 'nUnits', 'occuType', 'origDate', 'maturityDate', 
        'firstPayDate', 'oriBalance', 'salePrice', 'appValue', 'productType', 'oriTerm', 'iniRate', 'backEndRatio', 
        'frontEndRatio', 'loanType', 'loanPurpose', 'payFreq', 'channel', 'buyDown', 'docType', 'pmi', 'convert',
        'poolInsurance', 'recourse', 'ltv', 'negAmort', 'armIndex', 'margin', 'perRateCap', 'perRateFloor', 'perPayCap',
        'perPayFloor', 'lifeRateCap', 'lifeRateFloor', 'rateResetFreq', 'payResetFreq', 'firsRateResetPer',
        'firstPayResetPer', 'fico', 'lien', 'grade', 'prepayPenaly', 'prepayPenalyTerm', 'firstRateResetCap', 'cltv', 'cbsa',
        'ioTerm', 'io', 'msa', 'altA', 'convertDate', 'highLTV', 'paidOffFlag', 'pointsPaid', 'armIndexTerm', 'iniInvestorCode',
        'collateralType', 'activeStatus', 'period', 'sold']
        
    def getFormula(self, regressor, predictors, cat):
        '''get the formula in logit regression'''
        str_reg = regressor[0]
        str_pred_notcat = " + ".join([item for item in predictors if item not in cat])
        #str_pred_cat = " + ".join([("C(" + item + ")") for item in cat if item in predictors])
        #str_pred_cat = " + ".join([item for item in cat if item in predictors])
        str_pred_cat = " + ".join([("C(" + item + ")") for item in cat if item in predictors])

        #return (str_reg + " ~ " + str_pred_notcat + " + C(" + str_pred_cat + ') - 1')
        #return (str_reg + " ~ " + str_pred_notcat)
        return (str_reg + " ~ " + str_pred_notcat + " + " + str_pred_cat) 

    def getNAs(self):
        self.na_counts = self.data.isnull().sum()/self.data.shape[0]
        self.na_counts.to_csv('results.txt')

        return None

    def preProcess_default(self):
        #dele = ['addData', 'zip', 'state', 'propType', 'nUnits', 'occuType', 'origDate', 'maturityDate', 
        #'firstPayDate', 'oriBalance', 'salePrice', 'appValue', 'productType', 'oriTerm', 'backEndRatio', 
        #'frontEndRatio', 'loanType', 'loanPurpose', 'payFreq', 'channel', 'buyDown', 'docType', 'pmi', 'convert',
        #'poolInsurance', 'recourse', 'negAmort', 'armIndex', 'margin', 'perRateCap', 'perRateFloor', 'perPayCap',
        #'perPayFloor', 'lifeRateCap', 'lifeRateFloor', 'rateResetFreq', 'payResetFreq', 'firsRateResetPer',
        #'firstPayResetPer', 'grade', 'prepayPenaly', 'prepayPenalyTerm', 'firstRateResetCap', 'cltv', 'cbsa',
        #'ioTerm', 'io', 'msa', 'altA', 'convertDate', 'highLTV', 'paidOffFlag', 'pointsPaid', 'armIndexTerm', 'iniInvestorCode',
        #'collateralType', 'activeStatus', 'period', 'sold']
        #dele = ['addData', 'zip', 'origDate', 'maturityDate', 'firstPayDate', 'frontEndRatio', 'pmi', 'recourse', 
        #'firstPayResetPer', 'firstRateResetCap', 'convertDate', 'highLTV', 'pointsPaid', 'armIndexTerm', 'period',
        #'altA', 'armIndex', 'backEndRatio', 'buyDown', 'channel', 'cltv', 'convert', 'docType', 'firstPayResetPer', 
        #'grade', 'highLTV', 'io', 'ioTerm', 'lifeRateCap', 'lifeRateFloor', 'loanPurpose', 'margin', 'msa', 'negAmort', 
        #'payResetFreq', 'perPayCap', 'perPayFloor', 'perRateCap', 'perRateFloor', 'pointsPaid', 'poolInsurance', 'prepayPenalyTerm', 
        #'rateResetFreq', 'recourse', 'salePrice']
        cat = ['propType', 'occuType', 'productType', 'loanType', 'loanPurpose', 'payFreq', 'channel', 'docType', 'armIndex', 
        'lien', 'grade', 'iniInvestorCode', 'collateralType', 'activeStatus']

        regressor = ['default']

        predictors = ['iniRate', 'loanType', 'channel']
        #predictors = ['appValue', 'oriBalance', 'iniRate', 'channel', 'loanType']
        #predictors = list(set(self.var) - set(dele))
        #self.post_data = self.data[predictors + regressor].dropna()

        regFormula = self.getFormula(regressor, predictors, cat)
        #self.data = self.data.convert_objects(convert_numeric=True)
        #self.post_data = self.post_data.convert_objects(convert_numeric=True)

        #return smf.logit(formula=regFormula, data=self.data, maxiter=50)
        #return smf.logit(formula=regFormula, data=self.data, missing='drop')
        return smf.logit(formula=regFormula, data=self.data, missing='drop')

    def regression_default(self):
        fit = self.preProcess_default()

        return fit.fit(maxiter=10000)
#---------------------------end: logit regression class---------------------------------#

#---------------------------start: write the results------------------------------------#
class writeResults():
    def __init__(self, file):
        self.file = open(file, "w")

    def write_results(self, results):
        self.file.write(results)
        self.file.write('\n')

        return None

    def close(self):
        self.file.close()

        return None
#-----------------------------end: write the results------------------------------------#

if __name__ == "__main__":
    data = pd.read_csv('/share/PI/giesecke/wei/CleanData/clean_Middle_Atlantic.txt', delimiter='\t', header='infer', low_memory=False)
    regression = logitRegression(data)
    regression.getNAs()

    #writer = writeResults('results.txt')
    #writer.write_results(regression.getNAs())
    #writer.close()


