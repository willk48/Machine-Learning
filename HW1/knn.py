import random
import csv
import math
from math import sqrt
import sys
from collections import Counter
from collections import OrderedDict

#Will Kennedy
#Psuedo for KNN:
#1. choose k
#2. for each point in test set
#       find the distance to all other points, put in list
#       sort list low to high
#       grab the first k elements
#       majority vote decides the label at this test point
#       iterate over all test points
#3. store correct and incorrect guesses to be formatted in output
#4. Output confusion matrix with specified formatting to csv
#pdf instructions: https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW1/CSCI373_Homework1.pdf
#Usage: python knn.py mnist100.csv E 1 0.75 12345
dataset = sys.argv[1]
dist_func = sys.argv[2]
k_num = int(sys.argv[3])
train_perc = float(sys.argv[4])
seed = int(sys.argv[5])

#load in the csv as a collection of instances. 
#The format of this is each element of the return list is a single instance and the first element 
#is the label and the second element is all the instance variables
def load(csv_in):
	return_lst = []
	out_lst = []
	with open(csv_in, newline='') as csvfile:
		reader = csv.reader(csvfile,delimiter=',')
		for row in reader:
			out_lst.append(row)
	for item in out_lst:
		return_lst.append([item[0],item[1:]])
	return return_lst

def skip_first(lst):
	temp = lst[0]
	lst.remove(temp)
	return temp

#uses our specified seed and shuffles the list before splitting it into the standard x_train,x_test,y_train,y_test
#x's are variables and y's are the labels
def split(lst):
	train=[]
	test=[]
	random.seed(seed)
	random.shuffle(lst)
	test = lst[round(len(lst)*train_perc):]
	train = lst[:round(len(lst)*train_perc)]
	return train, test

def high_score(lst):
    data = Counter(lst)
    return data.most_common(1)

def knn(test_p, train,dist_func,k_num):
	if dist_func == 'E':
		workL = []
		indicies = []
		out = []
		for i in range(len(train)):
			workL.append(euc(test_p, train[i]))
		for i in range(k_num):
			curr_min = min(workL)
			indicies.append(workL.index(curr_min))
			workL.remove(curr_min)
		for i in range(len(indicies)):
			out.append(train[indicies[i]][0])
		if high_score(out)[0][1] == 1:
			return random.choice(out)
		else:
			return high_score(out)[0][0]
	else:
		workL = []
		indicies = []
		out = []
		for i in range(len(train)):
			workL.append(hamm(test_p, train[i]))
		for i in range(k_num):
			curr_min = min(workL)
			indicies.append(workL.index(curr_min))
			workL.remove(curr_min)
		for i in range(len(indicies)):
			out.append(train[indicies[i]][0])
		if high_score(out)[0][1] == 1:
			return random.choice(out)
		else:
			return high_score(out)[0][0]

def euc(test_p, train_p):
	dist = 0.0
	for i in range(0,len(test_p[1])):
		dist+=(float(test_p[1][i]) - float(train_p[1][i])) ** 2
	return math.sqrt(dist)

def hamm(test_p, train_p):
	dist = 0.0
	for i in range(0,len(test_p[1])):
		if test_p[1][i] != train_p[1][i]:
			dist+=1
	return dist

def gen_matrix(labels, pred, test):
	mat = []
	for i in range(len(labels)):
		mat.append([])
		for j in range(len(labels)):
			mat[i].append(0)
	for g in range(len(test)):
		mat[labels.index(pred[g])][labels.index(test[g][0])] +=1
	'''go through each item in each list and put a 1 in the matrix at the coordinates (labels.index(item)labels.index(item),)
	   example - pred: one, actual: seven
	   mat[1][7] = 1'''
	return mat

def output_matrix(mat, labels):
	with open(f'results_{dataset.split(".")[0]}_{dist_func}_{k_num}_{seed}.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(labels)
		for label in labels:
			row = [mat[labels.index(label)][labels.index(pred_label)] for pred_label in labels]
			writer.writerow(row + [label])

def main():
	data = load(dataset)
	first_row = skip_first(data)
	temp = [item[0] for item in data]
	labels = list(OrderedDict.fromkeys(temp))
	train,test = split(data)
	predictions = []
	for item in test:
		predictions.append(knn(item,train,dist_func,k_num))
	mat = gen_matrix(labels, predictions,test)
	output_matrix(mat,labels)

if __name__ == '__main__':
    main()