[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Ep4kCi5E)
# hw1-knn
# HW1: k-Nearest Neighbors
Name: Will Kennedy

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW1/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW1/)

# Data Sets 

This assignment contains four data sets which are based on three publicly available benchmarks:

1. monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape  jacket_color = red, then yes, else no. Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2. penguins.csv: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information (https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository (https://github.com/allisonhorst/palmerpenguins).

3. mnist100.csv: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 100 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: http://yann.lecun.com/exdb/mnist/.  It was converted to CSV file using the python code provided at: https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html

4. mnist1000.csv: The same as mnist100, except containing 1000 instances of each handwritten numeric digit.

# Research Questions

1. Pick a single random seed and a single training set percentage (document both in your README.md) and run k-Nearest Neighbors with a k = 1 on each of the four data sets first with the Hamming distance function.  What is the accuracy you observed on each data set?  

Seed: 1924
Training Percentage: 70% or 0.7

mnist100: 0.6740740741 or 67.41%

mnist1000: 0.7473333333 or 74.73%

monks1: 0.7384615385 or 74.85%

penguins: 0.6601941748 or 66.02%

Then, rerun k-Nearest Neighbors with the same seed, training set percentage, and k = 1 on only the penguins, mnist100, and mnist1000 datasets using the Euclidian distance function.  What is the accuracy you observed on each data set?  How do your accuracies compare between using the Hamming vs. Euclidian distance functions?

mnist100: 0.8333333333 or 83.33%

mnist1000: 0.9406666667 or 94.07%

penguins: 0.8155339806 or 81.55%

Accuracies tended to be higher using Euclidian distance over Manhattan distance. This is expected and makes sense the way we talked about these distance functions in class. I can see why we need Hamming distance in case attribute values are nominal, but it seems most times for small time complexity increases we can get more performance using euclidian distance.

2. Using the accuracies from Question 1, calculate a 95% confidence interval around each accuracy.  Show your arithmetic that you used to calculate the intervals.

SE = sqrt((acc*(1-acc))/n)
CI_95 = acc +- (1.96 * SE)

mnist100_H: 0.6740740741 +- (1.96* sqrt((0.6740740741*(1-0.6740740741))/270))
            0.6740740741 +- 0.05590973775170768
         CI=[0.6181643363482923,0.7299838118517077]

mnist1000_H: 0.7473333333 +- (1.96* sqrt((0.7473333333*(1-0.7473333333))/3000))
            0.7473333333 +- 0.015549864043909758
         CI=[0.7317834692560902,0.7628831973439097]

monks1_H: 0.7384615385 +- (1.96* sqrt((0.7384615385*(1-0.7384615385))/130))
            0.7384615385 +- 0.07554680776391384
         CI=[0.6629147307360861,0.8140083462639138]

penguins_H: 0.6601941748 +- (1.96* sqrt((0.6601941748*(1-0.6601941748))/103))
            0.6601941748 +- 0.09147211330284628
         CI=[0.5687220614971537,0.7516662881028463]

mnist100_E: 0.8333333333 +- (1.96* sqrt((0.8333333333*(1-0.8333333333))/300))
            0.8333333333 +- 0.042172485328743445
         CI=[0.7911608479712566,0.8755058186287434]

mnist1000_E: 0.9406666667 +- (1.96* sqrt((0.9406666667*(1-0.9406666667))/3000))
            0.9406666667 +- 0.00845400878957626
         CI=[0.9322126579104237,0.9491206754895762]

penguins_E: 0.8155339806 +- (1.96* sqrt((0.8155339806*(1-0.8155339806))/103))
            0.8155339806 +- 0.07490601233885132
         CI=[0.7406279682611487,0.8904399929388513]

3. How did your accuracy compare between the mnist100 and mnist1000 data sets when using the Euclidian distance function?  Which had the higher average?  Why do you think you observed this result?  Did their confidence intervals overlap?  What conclusion can we draw based on their confidence intervals?

mnist100_H: [0.6181643363482923,0.7299838118517077]
mnist1000_H: [0.7317834692560902,0.7628831973439097]
mnist100_E: [0.7911608479712566,0.8755058186287434]
mnist1000_E: [0.9322126579104237,0.9491206754895762]

On the mnist100 and mnist1000 datasets Euclidian distance performed better on our test data. In both cases the better model's 95% confidence interval is not contained in the 95% CI for the worse model. Therefore, the Euclidian distance function statistically significantly improves on the hamming distance model. I assume we acheived this result because of the way Euclidian distance punishes large swings in the data by squaring and taking the root. This way we are really only concerned with values that are similar across test instances and we want to punish high distances. Based on these confidence intervals it is best ot use Euclidian distance where available. 

4. Pick one data set and three different values of k (document both in your README.md).  Run the program with each value of k on that data set and compare the accuracy values observed.  Did changing the value of k have much of an effect on your results?  Speculate as to why or why not that observation occurred.

Datset: penguins.csv
Distance_funct: E
k = [1, 3, 5]
Seed: 5432
Training Percentage: 70% or 0.7

penguins_E_1: 0.8349514563
penguins_E_3: 0.6796116505
penguins_E_5: 0.7572815534

Increasing the value of k on the penguins dataset yielded lower accurarcy results at every increase of k. The best accuracy was observed with at k = 1 with lower accuracies at k =3 and k = 5. I think this occured becuase this is a relatively small and simple dataset. The community norm seems to be to start at k = sqrt(n_features), so it appears that we should have started at 2 on the penguins dataset. This is because at low feature counts we only want the first closest or the few first closest options. It appears that majority label on the closest point is the most predicitive in the knn structure, and most times lower values of k will be better while there are not many features.

## Bonus Question (Optional)

Did not answer optional question

5. Pick 10 different random seeds (document them in your README.md file) and rerun k-Nearest Neighbors with a k = 1 on the penguins.csv data.  Record the average of the accuracy across the 10 runs.

Next, rerun the program on the same 10 seeds but only consider two attributes at a time (ignoring the other two attributes not in the chosen pair).  Record the average accuracy for each pair of  attributes across the 10 seeds.  Since there are four attributes, there are six possible pairs of attributes (e.g., bill_length_mm-bill_depth_mm is one pair, so flipper_length_mm and body_mass_g would be ignored for this pair).

Finally, compare the average accuracy results between (1-6) all six pairs of attributes and (7) the results using all four attributes.  Did any pairs of attributes do as well (or better) than learning using all four attributes?  Speculate why you observed your results.
 
# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?
    I enjoyed finally cracking the code after spending a long time scratching my head trying to go about this assignment. I have not had to program anything without the use of external libraries in awhile so it was a good challenge for me to work with a matrix without pandas. 

2. Approximately how much time did you spend on this assignment?
    Approximately 7 hours

3. Did you adhere to the Honor Code?
    I have adhered to the honor code in this assessment.
    Will Kennedy
