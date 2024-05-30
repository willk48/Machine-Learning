[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/42rKmsKG)
# HW2: Decision Trees and Random Forests
Name: Will Kennedy

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW2/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW2/)

# Data Sets 

This assignment contains five data sets that are based on publicly available benchmarks:

1. **banknotes.csv**: A data set describing observed measurements about banknotes (i.e., cash) under an industrial print inspection camera.  The task in this data set is to predict whether a given bank note is authentic or a forgery.  The four attributes are each continuous measurements.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/banknote+authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

2. **mnist1000.csv**: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 1,000 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  It was converted to CSV file using the python code provided at: [https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html](https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html)

3. **occupancy.csv**: A data set of measurements describing a room in a building for a Smart Home application.  The task in this data set is to predict whether or not the room is occupied by people.  Each of the five attributes are continuous measurements.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

4. **penguins.csv**: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information [https://twitter.com/allison_horst/status/1270046399418138625](https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

5. **seismic.csv**: A data set of measurements describing seismic activity in the earth, measured from a wall in a Polish coal mine.  The task in this data set is to predict whether there will be a high energy seismic event within the next 8 hours.  The 18 attributes have a mix of types of values: 4 are ordinal attributes, and the other 14 are continuous.  The label is “no event” if there was no high energy seismic event in the next 8 hours, and “event” if there was such an event.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/seismic-bumps](https://archive.ics.uci.edu/ml/datasets/seismic-bumps)

# Research Questions

Please answer the following research questions using the `trees.py` program that you created.  Show your work for all calculations.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train a single tree for each of the five data sets.  

Seed: 1248
Training Percentage: 73

a. What is the accuracy of the test set predictions of each tree that you learned for each of the five data sets?

- banknotes: 193+168/193+168+7+3 = 361/371 or approx. 97.30% accuracy

- mnist1000: 2116/2116+584 = 2116/2700 or approx 78.37% accuracy

- occupancy: 5511/5511+42 = 5511/5552 or approx 99.26% accuracy

- penguins: 92/93 or approx. 98.92% accuracy

- seismic: 616/697 or approx. 88.38% accuracy

b. Calculate the 95% confidence interval around each accuracy in your answer to Question 1a.

SE = sqrt((acc*(1-acc))/n)
Z_.95 = 1.96, taken directly from slides, slightly confused on its origins. Are we meant to be calculating this?
95% interval is acc. +- 1.96*SE 

Checked with
from statsmodels.stats.proportion import proportion_confint
lower, upper = proportion_confint(correct_pred, total_possible, percent_interval = 0.05)
print('lower=%.3f, upper=%.3f' % (lower, upper))

- banknotes: SE = sqrt((361/371*(1-361/371))/371)=0.008408000648759607*1.96\
             CI = 0.9730458221024259 +- 0.008408000648759607*1.96 = 0.9730458221024259 +- 0.01647968127156883\
             CI = [0.956566140830857,0.9895255033739947]\
             checked_answ = [lower=0.957, upper=0.990]

- mnist1000: SE = sqrt((2116/2700*(1-2116/2700))/2700)=0.00792352821956372\
             CI = 0.7837037037037037 +- 0.00792352821956372*1.96 = 0.7837037037037037 +- 0.015530115310344893\
             CI = [0.7681735883933588,0.7992338190140486]\
             checked_answ = [lower=0.768, upper=0.799]

- occupancy: SE = sqrt((5511/5552*(1-5511/5552))/5552)=0.001149034179959971\
             CI = 0.9926152737752162 +- 0.001149034179959971*1.96 = 0.9926152737752162 +- 0.002252106992721543\
             CI = [0.9903631667824947,0.9948673807679377]\
             checked_answ = [lower=0.990, upper=0.995]

- penguins: SE = sqrt((92/93*(1-92/93))/93)=0.010694721775786493\
            0.989247311827957
            CI = 0.989247311827957 +- 0.010694721775786493*1.96 = 0.989247311827957 +- 0.020961654680541526\
            CI = [0.9682856571474154,1.0]?\
            Not sure if this is supposed to be bound by 1.0 but the math turns out a number higher than possible, trimmed to 1 for this answer. Confirmed by the statmodel function for CIs\
             checked_answ = [lower=0.968, upper=1.000]

- seismic: SE = sqrt((616/697*(1-616/697))/697)=0.01213902199623588\
           CI = 0.8837876614060258 +- 0.01213902199623588*1.96 = 0.8837876614060258 +- 0.023792483112622324\
           CI = [0.8599951782934036,0.9075801445186481]\
             checked_answ = [lower=0.860, upper=0.908]

#### Question 2

Using your program, visualize the tree for the penguins.csv data set learned in Question 1a.  Upload the tree visualization to GitHub as an image file. 

a. What rules were learned by this tree?

The tree seemed to learn that flipper length was the most predictive first attribute to check, seperating the data by flipper_length <=202.5. Having a flipper length higher than 202.5 seems to strongly suggest a penguin is a Gentoo. From that node the tree only has to check whether or not the bill depth is <= 17.65. If it is less than, it is definitely a gentoo. If not, check if the bill length is less than 46.55. If it is the penguin is an Adelie, if not - we have a Chinstrap. On the other side of the tree on flipper lengths actually less than 202.5 it is a little more tricky, but from here on out we can only either predict Adelie or Chinstrap. We have 4 more rules to check from this node. First if bill length > 43.35, it is likely a Chinstrap. The only way it would not be a Chinstrap is if the bill length was actually less than 45.9 (which would make it an Adelie). Now if bill length <= 43.55 but bill depth is > 16.7, it is an Adelie unless the bill depth is greater than 16.7 but less than 17.45 (then it is a Chinstrap). Finally, if the bill depth is <= 16.7 it is likely an Adelie unless the bill length is greater than 39.5, in which case it is a Chinstrap.  

b. How do these rules relate to your analysis in Lab 2 when you compared the average values of each attribute for each label (e.g., how the average bill length differed between the three species)?

These rules relate 1:1 with the findings from lab2. In lab 2 we showed that bill length and bill depth had the closest means across all species, so it makes complete sense that we had to check those values several times to be sure. On the other hand, the model found that flipper length was a good early predictor of Gentoo penguins, and that was something that did not come up in our lab2 testing. 

#### Question 3

Use a seed of 1234 and a training set percentage of 75%. Train a single tree to classify only the mnist1000.csv data set. 

a. Calculate the recall for your tree for each label.

recall = true positive/(true positive + false negative)

- zero: 165/(165+=Div) = 165/165+9+8+10+9+3+7+6+8+4 = 165/229 = 0.7205240175

At this point I just created a google sheet of the csv and processed the calculations. I will include this as the showing of my work in the repo.

- one: 0.7168458781

- two: 0.7642276423

- three: 0.7336065574

- four: 0.9111111111

- five: 0.8600823045

- six: 0.8605577689

- seven: 0.7094017094

- eight: 0.7550200803

- nine: 0.8549019608

b. Which label had the *lowest* recall?  Why do you think this label had the lowest?

Seven had the lowest recall. It looks like this is because we classified several of the '7's as '1's, leading us to miss out on several true sevens. It is easy to see why some 7s may look like 1s.

c. Which label had the *highest* recall? Why do you think this label had the highest?

Four had the highest recall. I assume this is becuase 4 has the distinct feature of not looking like any other number in most fonts. Most digits share a top half or bottom half with another number, and 7 and 1 tend to look similar in most peoples handwriting. Hence, why some people include a line through the middle of their sevens.

#### Question 4

Use a seed of 1234 and a training set percentage of 75%. Train a single tree to classify only the seismic.csv data set.

a. Calculate the recall for your tree for each label.

- event: 556/606 = 0.917491749

- no event: 4/40 = 0.10

b. What do you think these recalls imply about the usefulness of your model?

This implies our model is not very useful at predicting whether or not an event will occur because it was trained on data that has too many 'yes' instances. When models are trained on instances with too little variance in the labels, the model will converge on 100% to the one that is most often selected. It appears that in our test data we had 606 instances of event and only 40 instances of no event, so it easy to see that our model overfit to the 'win the game'.

c. Based on the data in the seismic.csv data set and the counts in your confusion matrix, why do you think this trend between the two recalls occurred?

These results imply that the model learned to just predict an event. Similar to class discussions about cancer research in this field. It appears our dataset may have too many instances of events proportionally, which leads the model to just select event and shoulder the cost of being wrong when there is none. 

#### Question 5

Using a seed of 1234, train a tree for each of the training percentages in [0.2, 0.4, 0.6, 0.8] with the mnist1000.csv, occupancy.csv, and penguins.csv data sets.  Plot the accuracies of the trees as a line chart, with training percentage as the x-axis, accuracy as the y-axis, and separate lines (in the same chart) for each data set.  Upload your line chart to GitHub as an image file.

a. What trends do you see for each data set?

- mnist1000: Mnist performed slightly better with each inscrease in training data, tapering off before reaching 80%. Trending upwards, but slowly. 

- occupancy: Occupancy quickly got to and maintained a ~98% accuracy throughout the train-test splits, so it showed no change.

- penguins: Penguins saw the most growth and progressed from ~84% to ~98% from 0.2 to 0.6. It stopped improving at this point, and stayed around 98.

b. Why do you think those trends occurred for each data set?

I think these trends can be explained by the properties of each dataset. Mnist did not perform well with this model at all so it makes sense that we see steady low improvement with most training data, because new training data is not helping all that much. This single tree will not converge on a solution to mnist. Occupany seemed to be the best suited for this classifier, and thus we see that even on train percentage 20% we are able to solve the problem. Penguins is another dataset suited for decisions trees, but this one required slightly more training data to understand the patterns becuase it has 1 more possible label than our binary occupancy task. 

#### Question 6

Using the same random seed and training percentage as Question 1, train a forest with 100 trees for each of the five data sets.

Seed: 1248
Training Percentage: 73

a. What is the accuracy of the test set predictions of each forest that you learned for each of the five data sets?

- banknotes:0.9919137466307277

- mnist1000:0.9425925925925925

- occupancy:0.9951368876080692

- penguins:0.989247311827957

- seismic:0.9312320916905444

b. Calculate the 95% confidence interval around each accuracy in your answer to Question 6a.

- banknotes:0.9919137466307277 +- 0.004649686502407919 = [lower=0.983, upper=1.000]

- mnist1000:0.9425925925925925 +- 0.004476762925339204 = [lower=0.934, upper=0.951]

- occupancy:0.9951368876080692 +- 0.0009336279336431593 = [lower=0.993, upper=0.997]

- penguins:0.989247311827957 +- 0.010694721775786493 = [lower=0.968, upper=1.000]

- seismic:0.9312320916905444 +- 0.009578426324514465 = [lower=0.912, upper=0.950]

#### Question 7

Compare the confidence intervals for each data set between Questions 1b and 6b.

a. For each data set, did any model (tree or forest) statistically significantly outperform the other?

- banknotes: 
    single tree: [lower=0.957, upper=0.990]
    100 trees: [lower=0.983, upper=1.000]

- mnist1000:
    single tree: [lower=0.768, upper=0.799]
    100 trees: [lower=0.934, upper=0.951]

- occupancy:
    single tree: [lower=0.990, upper=0.995]
    100 trees: [lower=0.993, upper=0.997]

- penguins:
    single tree: [lower=0.968, upper=1.000]
    100 trees: [lower=0.968, upper=1.000]

- seismic:
    single tree: [lower=0.860, upper=0.908]
    100 trees: [lower=0.912, upper=0.950]

Every dataset improved moving from 1 tree to a 100 tree forest. In the way we spoke about CIs in class where a model is statistically signifcatnly better than another model, the model would have to have a 95% CI that is not contained within the other. That is, the CI of the better model would entirely fall higher than the CI for the lesser model. By that criteria, 100t forests were statistically significantly better than a single tree on the datasets mnist1000 and seismic. By the same criteria we saw no significant improvement in penguins, occupancy, and banknotes. 

b. Based on your answer to Question 7a, what conclusions can you draw about the usefulness of training more than one tree in a random forest?

We can say that training more than 1 tree can be useful if the model did not converge or acheive the desired accuracy on 1 single tree. A single decision tree classifier performed well on most datasets (3/5), and we only saw statistically significatn improvement from datasets that produced sub 90% accuracy with a single tree. All initially well performing models did not see significant improvements moving from 1 tree to 100. 

#### Question 8

Using a seed of 1234, a training percentage of 0.75, and each of the numbers of trees in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], train a random forest for only the mnist1000.csv data set.  Plot the accuracies of the forest as a line chart, with number of trees as the x-axis and accuracy as the y-axis.  Upload your line chart to GitHub as an image file.

a. What trend do you observe?

We observe a general trend to be more accurate as the number of trees increases. It is interesting to see that the models did not get uniformly better at a relatively low tree num climb of 10 additional trees per model. I would have expected the next model to improve upon the previous across the board, but it appears that above 0.93 accuracy we see only small improvements and they are not uniform. That being said, the model was still improving up to 100 trees albiet slowly. So, it is unconfirmed where the actual limit of this classifier is on the problem. 

b. Why do you think this trend occurred for this data set?

It makes sense logically that if our model is converging at around ~93% or ~94% on the mnist dataset we would see non-uniform changes in accuracy for each new addition of more trees because our data is random and there is no way of knowing exactly how each forest will configure itself. I assume that we are just nearing the limit of what this variety of decision tree classifier can do with these parameters. The accuracy was still improving over larger changes in the number of trees (steps higher than 10), so it may be useful to give this problem even more trees. 
 
# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?

I found this assingment to be a big step up from HW1 in terms of usefulness. I feel like we are much closer to the kinds of machine learning problems that are out there in the world with this in comparison to HW1. The addition of libraries made the questions and analysis much more in depth and it allowed me to really see where this model shines after spending the time to create it. It was difficult to learn to use the graphing functions, but this is likely just because I am behind on labs.

2. Approximately how much time did you spend on this assignment?

I spent about 8 hours total on this assignment.

3. Did you adhere to the Honor Code?

I have adhered to the honor code in this assessment. 
Will Kennedy
