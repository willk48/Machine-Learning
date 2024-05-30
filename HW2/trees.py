#Will Kennedy
#imports
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
import matplotlib.pyplot as plt

#copied from assignment description, image generator for visualization
def log_tree(tree, dataset, dataset_filename, train_percentage, seed):
    # create the filename
    filename = ("tree"
                + "_" + dataset_filename[:-4]
                + "_1t"
                + "_" + str(int(train_percentage * 100)) + "p"
                + "_" + str(seed) + ".png")

    attributes = list(dataset.drop("label", axis=1))
    labels = sorted(list(dataset["label"].unique()))

    fig = plt.figure(figsize=(100, 100))
    plotted = sklearn.tree.plot_tree(tree,
                                     feature_names=attributes,
                                     class_names=labels,
                                     filled=True,
                                     rounded=True)
    fig.savefig(filename)

#read in command line arguments
data = sys.argv[1]
num_tree = int(sys.argv[2])
train_perc = float(sys.argv[3])
seed = int(sys.argv[4])

#load csv with pandas
def load(inp):
    return pd.read_csv(inp)

#split the dataset, using new pandas and skikit tools
def split(dataset, train_perc, seed):
    return train_test_split(dataset.iloc[:,1:], dataset.iloc[:, 0], train_size=train_perc, random_state=seed)

#main tree function
def tree_time(num_tree, x_train, x_test, y_train, y_test, dataset, data, train_perc, seed):
    if num_tree==1:
        tree = DecisionTreeClassifier(random_state=seed)
        tree.fit(x_train, y_train)
        #log_tree(tree, dataset, data, train_perc, seed)
        
    else:
        tree = RandomForestClassifier(n_estimators=num_tree,random_state=seed)
        tree.fit(x_train, y_train)
    #to enable the printing of the accuracy at every run
    #print(tree.score(x_test,y_test))
    return tree.predict(x_test)

#generate the output matrix in the form: results_<DataSet>_<NumTrees>t_<TrainingPercentage>p_<Seed>.csv
def gen_matrix(y_test, out, num_tree, dataset, data, train_perc, seed):
    mat = confusion_matrix(y_test, out)
    
    labels=dataset.iloc[:, 0].tolist()
    workList=[]
    for i in labels:
        if i not in workList:
            workList.append(i)
    
    frame = pd.DataFrame(mat, workList, workList)
    new_col = frame.columns.to_list()
    frame[' '] = new_col
    #print(frame.columns.to_list())
    #frame2 = frame.assign(frame.columns.to_list())
    frame.to_csv(f'results_{data.split(".")[0]}_{num_tree}t_{train_perc}p_{seed}.csv',index=False)

#main
def main():
    dataset=load(data)
    x_train, x_test, y_train, y_test = split(dataset,train_perc,seed)
    out = tree_time(num_tree,x_train,x_test,y_train,y_test,dataset,data,train_perc,seed)
    gen_matrix(y_test,out,num_tree,dataset,data,train_perc,seed)

if __name__ == "__main__":
    main()