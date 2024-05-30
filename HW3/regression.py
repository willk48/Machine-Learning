#Will Kennedy
#imports, some maybe not needed, copied from HW2 plus additions
import math
import random
import csv
import sys
import pandas as pd
import numpy as np
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

#read in command line arguments
data = sys.argv[1]
train_perc = float(sys.argv[2])
seed = int(sys.argv[3])
max_min_bool = sys.argv[4]

#helper function for one-hot encoding
def one_hot(orig, target_col):
    dummies = pd.get_dummies(orig[target_col])
    new = pd.concat([orig,dummies],axis=1)
    new = new.drop(target_col,axis=1)
    return new

#helper for max-min
def max_min(df, target_col):
    return (df[target_col]-df[target_col].min())/(df[target_col].max()-df[target_col].min())
    
#load csv with pandas, perform one-hot if req
def load(inp):
    frame = pd.read_csv(inp)
    new_frame = frame.copy()
    count = 0
    for col in frame.columns:
        if count != 0:
            if frame[col].dtype == 'object':
                #if column is categorical
                new_frame = one_hot(frame, col)
            elif max_min_bool == 'true' or max_min_bool == 'True':
                #if max-min selected, normalize data
                new_frame[col] = max_min(new_frame,col)
        frame = new_frame
        count = count+1
    return frame

#from HW2, split
def split(dataset, train_perc, seed):
    return train_test_split(dataset.iloc[:,1:], dataset.iloc[:, 0], train_size=train_perc, random_state=seed)

def main():
    #load and split data
    dataset=load(data)
    x_train, x_test, y_train, y_test = split(dataset,train_perc,seed)
    #train each model
    lin_reg = LinearRegression().fit(x_train,y_train)
    lasso = linear_model.Lasso().fit(x_train,y_train)
    ridge = linear_model.Ridge().fit(x_train,y_train)
    svm_2 = SVR(kernel='poly', degree=2)
    svm_2.fit(x_train, y_train)
    svm_3 = SVR(kernel='poly', degree=3)
    svm_3.fit(x_train, y_train)
    svm_4 = SVR(kernel='poly', degree=4)
    svm_4.fit(x_train, y_train)
    svm_poly = SVR(kernel='rbf')
    svm_poly.fit(x_train, y_train)
    tree = DecisionTreeRegressor(random_state=seed)
    tree.fit(x_train, y_train)
    
    #predictions based on training data
    lin_pred = lin_reg.predict(x_test)
    lin_mae = mean_absolute_error(y_test,lin_pred)
    lasso_pred = lasso.predict(x_test)
    lasso_mae = mean_absolute_error(y_test, lasso_pred)
    ridge_pred=ridge.predict(x_test)
    ridge_mae=mean_absolute_error(y_test,ridge_pred)
    svm2_pred=svm_2.predict(x_test)
    svm2_mae=mean_absolute_error(y_test,svm2_pred)
    svm3_pred=svm_3.predict(x_test)
    svm3_mae=mean_absolute_error(y_test,svm3_pred)
    svm4_pred=svm_4.predict(x_test)
    svm4_mae=mean_absolute_error(y_test,svm4_pred)
    svm_poly_pred=svm_poly.predict(x_test)
    svm_poly_mae=mean_absolute_error(y_test,svm_poly_pred)
    tree_pred = tree.predict(x_test)
    tree_mae = mean_absolute_error(y_test,tree_pred)

    #output formatting
    names=['LASSO','linear','ridge','svm_poly2','svm_poly3','svm_poly4','svm_rbf','tree']
    total_mae = [lasso_mae, lin_mae, ridge_mae, svm2_mae, svm3_mae, svm4_mae, svm_poly_mae, tree_mae]
    df=pd.DataFrame(total_mae)
    df2=pd.DataFrame(names,columns=['Model'])
    df2.insert(1,'MAE',total_mae,True)
    if max_min_bool == 'true' or max_min_bool == 'True':
        df2.to_csv(f'results_{data.split(".")[0]}_{train_perc}p_{seed}_rescaled.csv',index=False)
    else:
        df2.to_csv(f'results_{data.split(".")[0]}_{train_perc}p_{seed}.csv',index=False)
    
    #bar_chart(names,total_mae,data)
    

    #line_data = [seoul_LASSO,seoul_linear,seoul_ridge,seoul_svm_poly2,seoul_svm_poly3,seoul_svm_poly4,seoul_svm_rbf,seoul_tree]
    #line_chart(line_data,[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],"seoulbike.csv")

#create bar chart with math plt lib
def bar_chart(names, data, filename):
    fig = plt.figure(figsize=(10,5))
    plt.bar(names, data, color='blue',width=0.5)
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title(filename)
    plt.savefig(f'{filename.split(".")[0]}.png')

#create line graph for Q3 and Q4
def line_chart(vals, train_percs, filename):
    # Create a line chart
    plt.figure(figsize=(10, 8))
    plt.plot(train_percs, vals[0], marker='o', linestyle='-', label= "LASSO")
    plt.plot(train_percs, vals[1], marker='o', linestyle='-', label= "Lin_Reg")
    plt.plot(train_percs, vals[2], marker='o', linestyle='-', label= "Ridge")
    plt.plot(train_percs, vals[3], marker='o', linestyle='-', label= "SVM_Poly2")
    plt.plot(train_percs, vals[4], marker='o', linestyle='-', label= "SVM_Poly3")
    plt.plot(train_percs, vals[5], marker='o', linestyle='-', label= "SVM_Poly4")
    plt.plot(train_percs, vals[6], marker='o', linestyle='-', label= "SVM_RBF")
    plt.plot(train_percs, vals[7], marker='o', linestyle='-', label= "Tree")
 
    plt.title(f'MAE Over Different Training Percentages - {filename}')
    plt.xlabel('Training Percentage')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{filename.split(".")[0]}_rescaled_line.png')

if __name__ == "__main__":
    main()