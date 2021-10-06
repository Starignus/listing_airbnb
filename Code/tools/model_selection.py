#!/usr/bin/env/python

"""
Script for model selection
"""

import numpy as np
import xgboost as xg
from datetime import datetime
import matplotlib.pyplot as plt


def grid_search(gridsearch_params,
                orignal_params,
                param_list_name,
                dtrain,
                num_boost_round,
                seed=42,
                nfold=5,
                metrics={'rmse'},
                early_stopping_rounds=10):
    """

    :param gridsearch_params:
    :param orignal_params:
    :param param_list_name:
    :param dtrain:
    :param num_boost_round:
    :param seed:
    :param nfold:
    :param metrics:
    :param early_stopping_rounds:
    :return:
    """
    # setting to abig numberhe initial best RMSE
    startTime = datetime.now()
    min_rmse = np.inf
    min_test_err = []
    min_train_err = []
    boost_rounds_list = []
    if isinstance(gridsearch_params[0], tuple):
        for param_1, param_2 in gridsearch_params:
            print("CV for {}={} and {}={}".format(param_list_name[0],
                                                  param_1, param_list_name[1], param_2))

            # update dictionary with parameters
            orignal_params[param_list_name[0]] = param_1
            orignal_params[param_list_name[1]] = param_2

            # CV
            boost_rounds, test_mean_rmse = xgb_cv_session(boost_rounds_list, dtrain, early_stopping_rounds, metrics,
                                                          min_test_err, min_train_err, nfold, num_boost_round,
                                                          orignal_params, seed)
            if test_mean_rmse < min_rmse:
                min_rmse = test_mean_rmse
                best_params = (param_1, param_2)
        print("Best params: {} = {}, {} = {}, MSER: {}".format(param_list_name[0],
                                                               best_params[0],
                                                               param_list_name[1],
                                                               best_params[1], min_rmse))
    else:
        for param in gridsearch_params:
            print("CV for {}={} ".format(param_list_name[0], param))

            # update dictionary with parameters
            orignal_params[param_list_name[0]] = param

            # CV
            boost_rounds, test_mean_rmse = xgb_cv_session(boost_rounds_list, dtrain, early_stopping_rounds, metrics,
                                                          min_test_err, min_train_err, nfold, num_boost_round,
                                                          orignal_params, seed)

            if test_mean_rmse < min_rmse:
                min_rmse = test_mean_rmse
                best_params = param
        print("Best param: {} MSER: {}".format(best_params, min_rmse))
    print('Execution Time', datetime.now() - startTime)
    return min_test_err, min_train_err, boost_rounds_list


def xgb_cv_session(boost_rounds_list, dtrain, early_stopping_rounds, metrics, min_test_err, min_train_err, nfold,
                   num_boost_round, orignal_params, seed):
    """

    :param boost_rounds_list:
    :param dtrain:
    :param early_stopping_rounds:
    :param metrics:
    :param min_test_err:
    :param min_train_err:
    :param nfold:
    :param num_boost_round:
    :param orignal_params:
    :param seed:
    :return:
    """
    cv_results = xg.cv(params=orignal_params,
                       dtrain=dtrain,
                       num_boost_round=num_boost_round,
                       seed=seed,
                       nfold=nfold,
                       metrics=metrics,
                       early_stopping_rounds=early_stopping_rounds)
    # Update best RMSE
    test_mean_rmse = cv_results['test-rmse-mean'].min()
    # it starts from zero so + 1
    boost_rounds = cv_results['test-rmse-mean'].argmin() + 1
    train_mean_rmse = cv_results['train-rmse-mean'].min()
    min_test_err.append(test_mean_rmse)
    min_train_err.append(train_mean_rmse)
    boost_rounds_list.append(boost_rounds)
    print("\RMSE {} for {} rounds".format(test_mean_rmse, boost_rounds))
    return boost_rounds, test_mean_rmse


def grid_search_rounds(min_test_err, min_train_err, boost_rounds_list, y_limit=None):
    boosters_mean = np.round(np.mean(boost_rounds_list))
    x_axis = range(0, len(min_test_err))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, np.array(min_train_err), label='Train')
    ax.plot(x_axis, np.array(min_train_err), label='Test')
    ax.legend()
    plt.ylabel('Mean Best CV RMSR')
    plt.title('XGBoost Regression with {} mean numer of boosters'.format(boosters_mean))
    plt.xlabel('Summary  of the Grid-Search-Rounds')
    if y_limit is not None:
        plt.ylim(y_limit)