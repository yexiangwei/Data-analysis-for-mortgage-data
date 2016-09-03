import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.formula.api as smf
import time
import tensorflow as tf
import random

if __name__ == "__main__":
    data = pd.read_csv('/share/PI/giesecke/wei/CleanData/clean_Middle_Atlantic.txt', delimiter='\t', header='infer', low_memory=False)

    # set up
    categorical = ['propType', 'occuType', 'productType', 'loanType', 'loanPurpose', 'payFreq', 'channel', 'docType', 'armIndex', 
        'lien', 'grade', 'iniInvestorCode', 'collateralType', 'activeStatus']

    var_names = ['addData', 'zip', 'state', 'propType', 'nUnits', 'occuType', 'origDate', 'maturityDate', 
        'firstPayDate', 'oriBalance', 'salePrice', 'appValue', 'productType', 'oriTerm', 'iniRate', 'backEndRatio', 
        'frontEndRatio', 'loanType', 'loanPurpose', 'payFreq', 'channel', 'buyDown', 'docType', 'pmi', 'convert',
        'poolInsurance', 'recourse', 'ltv', 'negAmort', 'armIndex', 'margin', 'perRateCap', 'perRateFloor', 'perPayCap',
        'perPayFloor', 'lifeRateCap', 'lifeRateFloor', 'rateResetFreq', 'payResetFreq', 'firsRateResetPer',
        'firstPayResetPer', 'fico', 'lien', 'grade', 'prepayPenaly', 'prepayPenalyTerm', 'firstRateResetCap', 'cltv', 'cbsa',
        'ioTerm', 'io', 'msa', 'altA', 'convertDate', 'highLTV', 'paidOffFlag', 'pointsPaid', 'armIndexTerm', 'iniInvestorCode',
        'collateralType', 'activeStatus', 'period']

    # Predictors
    var_used = ['iniRate', 'lien', 'propType', 'appValue', 'armIndex', 'buyDown', 'cbsa', 'channel', 'collateralType', 'docType', 
            'fico', 'loanType', 'ltv', 'grade']
    var_argu = []

    # Add dummies for NAs
    for it_var in var_used:
        null_ind = data[it_var].isnull()
        if (null_ind.sum() != 0):
            mis_dummy = it_var + '_missing'#name
            if it_var not in categorical:
                var_argu += [it_var, mis_dummy]
                data.loc[:, mis_dummy] = 0.
                data.loc[null_ind, mis_dummy] = 1.
                data.loc[null_ind, it_var] = 0.
            else:
                data.loc[null_ind, it_var] = mis_dummy

    data_predictors = data[var_argu]

    for it_var in var_used:
        if it_var in categorical:
            dummy_mat = pd.get_dummies(data[it_var], prefix=it_var).iloc[:, 1:]
            var_argu += list(dummy_mat.columns.values)
            data_predictors[list(dummy_mat.columns.values)] = dummy_mat

    # Regressors
    data.loc[:, 'default_no'] = 1.
    data.loc[:, 'default_no'] = data.loc[:, 'default_no'] - data.loc[:, 'default']
    data_regressors = data[['default', 'default_no']]

    # TensorFlow
    learning_rate = 1e-17
    batch_size = 50000
    training_epochs = 100
    display_step = 1
    numObs = data_predictors.shape[0]
    numPs = data_predictors.shape[1]

    x = tf.placeholder("float", [None, numPs])

    # Set model weights
    W = tf.Variable(tf.zeros([numPs, 2]))
    b = tf.Variable(tf.zeros([2]))

    y = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    y_ = tf.placeholder(tf.float32, [None, 2])

    cost = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.initialize_all_variables()

    predictors = data_predictors.as_matrix()
    regressor = data_regressors.as_matrix()

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(numObs/batch_size)

            for i in range(total_batch):
                batch_id = random.sample(xrange(numObs), batch_size)

                batch_xs, batch_ys = predictors[batch_id], regressor[batch_id]
                opt = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})/total_batch
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print W.eval(), b.eval()


