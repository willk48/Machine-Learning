#Will Kennedy
#imports, some maybe not needed, copied from HW2\3 plus additions
import math
import random
import csv
import sys
import pandas as pd
import numpy as np
import sklearn.tree
import tensorflow as tf
from keras.losses import mean_absolute_error
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
learn_rate=float(sys.argv[2])
hidden_neurons = int(sys.argv[3])
train_perc = float(sys.argv[4])
seed = int(sys.argv[5])

#so I can reuse my old code
max_min_bool = "True"

#helper function for one-hot encoding
def one_hot(orig, target_col):
    dummies = pd.get_dummies(orig[target_col])
    new = pd.concat([orig,dummies],axis=1)
    new = new.drop(target_col,axis=1)
    return new

#helper for max-min
def max_min(df, target_col):
    return (df[target_col]-df[target_col].min())/(df[target_col].max()-df[target_col].min())

#from lab 5, map categorical labels to ints
def convert_labels(dataset):
    # create a dictionary mapping 
    numbers = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,"Adelie":0,"Chinstrap":1,"Gentoo":2}
    # convert each of the string labels into the corresponding number
    for name in numbers:
        number = numbers[name]
        dataset.loc[dataset["label"] == name, "label"] = number

#load csv with pandas, perform one-hot if req
def load(inp):
    frame = pd.read_csv(inp)
    new_frame = frame.__deepcopy__()
    count = 0
    for col in frame.columns:
        if count != 0:
            if frame[col].dtype == 'object':
                #if column is categorical
                new_frame = one_hot(frame, col)
            elif (frame[col] != 0).any():#normalize data
                new_frame[col] = max_min(new_frame,col)
        frame = new_frame
        count = count+1
    return frame

#from HW2, split
def split(dataset, train_perc, seed):
    return train_test_split(dataset.iloc[:,1:], dataset.iloc[:, 0], train_size=train_perc, random_state=seed)

# creates a neural network with one hidden layer, copied from lab 5, specifying as classification problem
def create_network_class(hidden_num,df):
    hidden_layer = tf.keras.layers.Dense(hidden_num, activation='sigmoid')
    '''NUMBER OF POSSIBLE LABELS OR 1 FOR REG'''
    output_layer = tf.keras.layers.Dense(df['label'].nunique()) # num unique on labels column for all possible labels
    print(df['label'].nunique())
    all_layers = [hidden_layer, output_layer]
    network = tf.keras.models.Sequential(all_layers)
    return network
# creates a neural network with one hidden layer, copied from lab 5, specifying as regression problem
def create_network_reg(hidden_num):
    hidden_layer = tf.keras.layers.Dense(hidden_num, activation='sigmoid')
    '''NUMBER OF POSSIBLE LABELS OR 1 FOR REG'''
    output_layer = tf.keras.layers.Dense(1) # 1 for regression task
    all_layers = [hidden_layer, output_layer]
    network = tf.keras.models.Sequential(all_layers)
    return network

# trains a neural network with given training data, from lab 5, specifying classification problem
def train_network_class(network, training_X, training_y,learning_rate,data):
    # create the algorithm that learns the weight of the network
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # create the loss function function that tells optimizer how much error it has in its predictions
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # prepare the network for training
    network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    # create a logger to save the training details to file
    csv_logger = tf.keras.callbacks.CSVLogger(f'{data.split(".")[0]}_training.csv')
    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])

# trains a neural network with given training data, from lab 5, specifying regression problem
def train_network_reg(network, training_X, training_y,learning_rate,data):
    # create the algorithm that learns the weight of the network
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # create the loss function function that tells optimizer how much error it has in its predictions
    loss_function = tf.keras.losses.MeanAbsoluteError()
    # prepare the network for training
    network.compile(optimizer=optimizer, loss=loss_function, metrics=["mean_absolute_error"])
    # create a logger to save the training details to file
    csv_logger = tf.keras.callbacks.CSVLogger(f'{data.split(".")[0]}_training.csv')
    # train the network for 200 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])

def main():
    if data == "mnist1000.csv" or data == "penguins.csv":
        #classification task, load instances and normalize+convert labels
        print("class")
        df1 = load(data)
        convert_labels(df1)
        #df1.to_csv("check_labels2.csv", index=False)
        x_train, x_test, y_train, y_test = split(df1,train_perc,seed)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        #create the network and train it
        net1 = create_network_class(hidden_neurons,df1)
        train_network_class(net1,x_train,y_train,learn_rate,data)

        eval = net1.evaluate(x_test,y_test)
        gen_out(data,learn_rate,hidden_neurons,eval[1])
    else:
        #regression task, load instances and normalize
        print("reg")
        df3 = load(data)
        x_train, x_test, y_train, y_test = split(df3,train_perc,seed)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        #y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        y_train = tf.convert_to_tensor(y_train)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        #y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        y_test = tf.convert_to_tensor(y_test)
        #create the network and train it
        net2 = create_network_reg(hidden_neurons)
        train_network_reg(net2,x_train,y_train,learn_rate,data)

        eval2 = net2.evaluate(x_test,y_test)
        gen_out(data,learn_rate,hidden_neurons,eval2[1])
    #gen_line_chart('seoulbike')
    #gen_line_chart_2('seoulbike_training')

#generates output in the specified format, in a results.csv
def gen_out(dataset, learn_rate, neurons, perf):
    #df = pd.DataFrame(dataset, learn_rate, neurons, perf)
    file1 = open('results.csv', 'a')
    file1.write(f'{dataset},{learn_rate},{neurons},{perf}')
    file1.write("\n")
    file1.close()

#create the results file if it doesnt exist, usage: add to main method at first run, then delete
#I could not figure out how to create the file if it does not exist and print the first line with
#the fields in this fashion.
def create_out_file():
    file = open('results.csv', 'w+')
    file.write("Dataset,LearningRate,Neurons,Performance")
    file.write("\n")
    file.close()

#generates line charts in the specified format
def gen_line_chart_1(data):
    df= pd.read_csv(f'{data}_res.csv')
    perf = df['Performance']
    neurons = df['Neurons']
    rates = df['LearningRate']
    
    line_5_df=df.loc[df['LearningRate'] == 1e-05]
    line_5=plt.plot(line_5_df['Neurons'],line_5_df['Performance'],label='0.00001')
    line_4_df=df.loc[df['LearningRate'] == 1e-04]
    line_4=plt.plot(line_4_df['Neurons'],line_4_df['Performance'],label='0.0001')
    line_3_df=df.loc[df['LearningRate'] == 1e-03]
    line_3=plt.plot(line_3_df['Neurons'],line_3_df['Performance'],label='0.001')
    line_2_df=df.loc[df['LearningRate'] == 0.01]
    line_2=plt.plot(line_2_df['Neurons'],line_2_df['Performance'],label='0.01')
    line_1_df=df.loc[df['LearningRate'] == 0.1]
    line_1=plt.plot(line_1_df['Neurons'],line_1_df['Performance'],label='0.1')
    
    leg = plt.legend(title="Learning_Rate")
    plt.grid(True)
    plt.ylabel('Performance')
    plt.xlabel('Hidden Neurons')
    plt.title(f'{data}.csv')
    plt.savefig(f'{data}_line_chart.png')

#generates line charts for question 4
def gen_line_chart_2(data):
    df1= pd.read_csv(f'{data}0.01.csv')
    #sort into new dfs
    epoch1 = df1['epoch']
    loss1 = df1['loss']
    mae1 = df1['mean_absolute_error']
    val_loss1 = df1['val_loss']
    val_mae1 = df1['val_mean_absolute_error']
    #second one
    df2= pd.read_csv(f'{data}0.00001.csv')
    epoch2 = df2['epoch']
    loss2 = df2['loss']
    mae2 = df2['mean_absolute_error']
    val_loss2 = df2['val_loss']
    val_mae2 = df2['val_mean_absolute_error']
    plt.clf()
    line_1=plt.plot(epoch1, mae1,label='mean_absolute_error',linewidth=5.0)
    line_2=plt.plot(epoch1, loss1,label='loss')
    line_4=plt.plot(epoch1, val_loss1,label='val_loss', linewidth=5.0)
    line_3=plt.plot(epoch1, val_mae1,label='val_mean_absolute_error')

    leg = plt.legend(title="Metric")
    plt.grid(True)
    plt.ylabel('MAE-Loss')
    plt.xlabel('Epoch')
    plt.title(f'{data}0.01.csv')
    plt.savefig(f'{data}_0.01.png')

    plt.clf()
    line_12=plt.plot(epoch2,mae2,label='mean_absolute_error',linewidth=5.0)
    line_22=plt.plot(epoch2, loss2,label='loss')
    line_42=plt.plot(epoch2, val_loss2,label='val_loss', linewidth=5.0)
    line_32=plt.plot(epoch2, val_mae2,label='val_mean_absolute_error')

    leg = plt.legend(title="Metric")
    plt.grid(True)
    plt.ylabel('MAE-Loss')
    plt.xlabel('Epoch')
    plt.title(f'{data}0.00001.csv')
    plt.savefig(f'{data}_0.00001.png')

if __name__ == "__main__":
    main()