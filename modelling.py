# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:10:38 2020

@author: ShambhaviChati
"""
import pickle
import datetime
import datetime
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from pyod.utils.utility import standardizer

from preparation import data_prep


def train_test_splitter(df):
    """
    Function inputs a datframe and splits into train and test sets.
    The splits are 80-20 respectively.
    Input: Main DataFrame. 
    Output: Training and testing DataFrames.
    """
    RANDOM_SEED=40
    train,test = train_test_split(df, test_size=0.2,train_size = 0.8,  random_state=RANDOM_SEED)
    print('train and test split done')
    return train, test


def training(train, train_copy, model_pickle_name):
    
    """
    Function for training the test data and calculating test scores. 
    Input: Training dataframe(encoded), copy of the training data, name of model pickle filw which will be used for testing.
    Output: Training scores as a DataFrame.
    """

    print("Training for IF")
    clf = IsolationForest( contamination=0.02, max_features=0.8, bootstrap= True, random_state = 90)
    clf.fit(train)
    filename = os.path.join('pickle_files', 'IF_'+model_pickle_name+'_if.sav')
    pickle.dump(clf, open(filename, 'wb'))

    # The scores are inversed. Higher scores suggest high probablity of anomaly
    scores = (1 - clf.decision_function(train))
    x = pd.Series(scores, name=" Train Scores")
    y_train_scores_IF = scores.tolist()
    
    train_scores = pd.DataFrame( {'IF_SCORES_train' : y_train_scores_IF,
                                 } )
    train_scores['IF_SCORES_train_INDEXED'] = train_scores['IF_SCORES_train']/train_scores['IF_SCORES_train'].max()
    print("Training complete")
    return train_scores

def missing_columns(df, df_test):
    """
    Function to sync columns between the training and test DataFrames.
    Input: Takes the train and test Dataframes.
    Output: Test DataFrame with same columns as the train.

    """
    missing_cols = set(df.columns) - set(df_test.columns)
# Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        df_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
    df_test = df_test[df.columns]
    return df_test


def scoring(test,  test_copy,  model_pickle_name):

    """
    Function to score the Test data.
    Input: The test DataFrame(encoded), copy of the test DataFrame and the model pickle name
    Output: Test Scores as DataFrame

    """    
    print("Scoring Started")  
    filename = os.path.join('pickle_files', 'IF_'+ model_pickle_name+ '_if.sav')
    c = pickle.load(open(filename, 'rb'))
    print('model used:', filename)

    # The scores are inversed. Higher scores suggest high probablity of anomaly.
    scores = (1 - c.decision_function(test))

    x = pd.Series(scores, name="Test Scores")
    y_test_scores_IF = scores.tolist()    
    
    test_scores = pd.DataFrame( {'IF_SCORES_test' : y_test_scores_IF,
                                 } )
    test_scores['IF_SCORES_test_INDEXED'] = test_scores['IF_SCORES_test']/test_scores['IF_SCORES_test'].max()
    print("Scoring Done")
    return  test_scores

def main():
    # Reading files and assigning scoring flag and pickle file name.   
    model_pickle_name = ' train'
    scoring_flag = 'N'
    #data = pd.read_excel("D:\Shambhavi Chati\Thinkrisk\Victoria\Creditcard\project_files\data\Credit Card Portal Extract - 20201013.xls")
    
    # Splitting the data as train and test sets
    train , test = train_test_splitter(data)
    data_final_train, data_final_train_copy, model_col_list_train = data_prep(data)
    data_final_test, data_final_test_copy, model_col_list_test = data_prep(test)
    
    # Syncing missing columns
    data_final_test = missing_columns(data_final_train, data_final_test)
    
    # evaluating Scores as per the Scoring Flag
    if scoring_flag == 'N':        
        
        train_scores = training(data_final_train, data_final_train_copy,  model_pickle_name )
        train_data_scored = pd.concat([train_scores, data_final_train_copy], axis = 1, join = 'outer', ignore_index=False, sort=False)
        #train_data_scored.to_csv("D:\Shambhavi Chati\Thinkrisk\Victoria\Creditcard\project_files\outputs\trained_data_scored.csv")
    
    if scoring_flag == 'Y':
        
        test_scores = scoring(data_final_test, data_final_test_copy, model_pickle_name)
        test_data_scored = pd.concat([test_scores, data_final_test_copy], axis = 1, join = 'outer', ignore_index=False, sort=False)

        #test_data_scored.to_csv("D:\\Shambhavi Chati\Thinkrisk\Victoria\Creditcard\project_files\outputs\test_data_scored.csv")
    
        
  
    

if __name__ =="__main__":
    main()
