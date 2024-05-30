[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/u5RbYg_2)
# HW4: Classification and Regression with Neural Networks
Name: Will Kennedy

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW4/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW4/)

# Data Sets 

This assignment contains four data sets that are based on publicly available benchmarks:

1. **mnist1000.csv**: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 1,000 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  It was converted to CSV file using the python code provided at: [https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html](https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html)

2. **penguins.csv**: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information [https://twitter.com/allison_horst/status/1270046399418138625](https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

3.	**energy.csv**: A data set describing the energy consumption in 10-minute increments by appliances in a low-energy residence in Belgium.  The task is to predict how much energy was consumed by appliances.  Each of the 27 attributes are numeric and describe measurements from sensors in the residence or nearby weather stations, as well as energy usage by lights.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

4.	**seoulbike.csv**: Another data set describing bike rentals in a metropolitan area (Seoul, South Korea).  Again, the task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 11 attributes are a mix of 2 categorical and 9 numeric attributes, including information such as the season, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

# Research Questions

Please answer the following research questions using the `neuralnet.py` program that you created.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train neural networks for the `penguins.csv` and `mnist1000.csv` data sets with each of the following numbers of hidden neurons `[8, 16, 32, 64, 128, 256]` and learning rates `[0.00001, 0.0001, 0.001, 0.01, 0.1]`.

Seed: 1513
Train_percentage: 0.75, 75%

a. Using the resulting accuracies from the 30 neural networks trained for each data set, create one **line chart** for each data set where the x-axis is the number of neurons, the y-axis is the accuracy of the neural network on the test set, and there are separate lines for each learning rate.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `penguins_line.png`).  Upload those two image files to your GitHub repository.

done

b. For each classification data set, what general trend do see as the number of neurons increases?  Did performance improve or worsen?  Why do you think this occurred?

Classification tasks got better as the number of neurons increased on both datasets. In mnist we can see increases in accuracy all the way up to neuron count 256 for every learning rate except for 0.1, which performed worse on higher neuron counts. In the penguins dataset we see a large gap between each learning rate, but all learning rate's performance increased as the neuron count increased. I think this general trend (increases in neuron count lead to better classification) is due to the fact that we have not yet overfit these datasets. We would expect to see declining test accuracies at every subsequent step beyond 100% validation accuracy, which may be what is happening at lower learning rates and causing the strange aforementioned behavior.

c. For each classification data set, what general trend do you see as the learning rate increases?  Did performance improve or worsen?  Why do you think this occurred?

The penguins dataset got better at every increase in learning rate peaking at 100% accuracy at 0.1 learn rate and 16 neurons. On the contrary, mnist got better up to and including 0.01, but performed poorly on 0.1. This was likely due to overfitting on validation data with the high rate of learning. As a general trend, both datasets yielded better results at higher learning rates. However, it will be important to test and note where any droppoff may occur, becuase it does not appear to be universal across datasets where we may overfit. 

#### Question 2

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train neural networks for the `energy.csv` and `seoulbike.csv` data sets with each of the following numbers of hidden neurons `[8, 16, 32, 64, 128, 256]` and learning rates `[0.00001, 0.0001, 0.001, 0.01, 0.1]`.

Seed: 93841
Training Percentage: 0.7, 70%

a. Using the resulting MAEs from the 30 neural networks trained for each data set, create one **line chart** for each data set where the x-axis is the number of neurons, the y-axis is the MAE of the neural network on the test set, and there are separate lines for each learning rate.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `energy_line.png`).  Upload those two image files to your GitHub repository.

Done

b. For each regression data set, what general trend do see as the number of neurons increases?  Did performance improve or worsen?  Why do you think this occurred?

For the first time we have a nice trend on every learning rate. Each learning rate improved as neuron count increased all the way to 256. Most slowed their rate of improvement after 128 neurons, but MAE was still decreasing at 256 neurons on every learning rate in both datasets. We would likely see a stagnation at some future neuron count that introduces overfitting, but as it stands more is better. 

c. For each regression data set, what general trend do you see as the learning rate increases?  Did performance improve or worsen?  Why do you think this occurred?

Similarly to the last question, we see a steady increase in performance all the way up to and including 0.1 on the smaller seoulbike dataset, but a notable drop off at 0.1 on the larger energy dataset. Energy improved up to and including 0.01 but showed worse performance on 0.1, which is likely due to overfitting on too steep a learning rate and specializing on the training and validation sets.

#### Question 3

a. Based on your answers to Questions 1 and 2 above, which do you think is more important to optimize first -- the learning rate or the number of neurons in the network?  Why is it more important to optimize that hyperparameter first?

**Learning rate** appears to offer far more immediate optimization benefits than the number of hidden neurons in the hidden layer. For smaller datasets like penguins and seoulbike, the best performing learning rate is performing more than 2 times better than the worst performing rate. Consequently, when the learning rate approaches the most desirable configuration we see a decline in the benefits of adding more neurons. All models definitley improved over some range of adding neurons, but the scale is much more modest. The number of hidden neurons appears to have the most benefit when the learning rate is actually improperly configured. Refer to line graphs for energy and mnist1000: on larger datasets the lowest learning rate of 0.00001 saw huge benefit to simply adding more neurons. Overall, learning rate appears to have a much larger impact.

b. Based on your answers to Questions 1 and 2 above, when might we want to start with a small learning rate (closer to 0)?  When might we want want to start with a larger learning rate (closer to 1)?

In all the cases enumerated for question 1 and 2, it appears to be best to start closer to 1. Both smaller datasets performed the best on the highest learning rate and of our 5 learning rate options, the second highest option at 0.01 was the best performing on the larger datasets. In this way we could say that larger datasets appear to require learning rates closer to 0 than their smaller counterparts. However, of the 5 possible learning rates the two closest to 1 were the best performing in all 4 cases (0.1 and 0.01).

c. Based on your answers to Questions 1 and 2 above, when might we want to start with a small number of neurons?  When might we want want to start with a larger number of neurons?

It appears that increasing the number of neurons has diminishing returns when the learning rate is close to optimal. So I would say with the data we surveyed, it appears to be best to start with 16 neurons and work your way up the ladder. The only real difference we see increasing neuron count is when the learning rate is pretty far off. That being said, the 8 neuron category did not perform well on most problems; so it may be better to omit it. I would say it would be safe to start with a larger number of neurons on large classification problems, which showed improvement quickly for most learning rates.

#### Question 4

a. Using the `penguins.csv` data set, a number of hidden neurons of `128`, a training percentage of `0.75`, and a random seed of `54321`, create two line charts that demonstrate the performance on the **training** and **validation** sets during training: one line chart for a learning rate of `0.01` and another for a learning rate of `0.00001`.  As in Lab 5, the x-axis should be the epoch, the y-axis should be the loss and accuracy, and there should be four lines -- one each for the `[loss, val_loss, accuracy, val_accuracy]` tracked by the `CSVLogger` during training.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `penguins_0.01.png`).

done

b. Similarly, using the `seoulbike.csv` data set, a number of hidden neurons of `128`, a training percentage of `0.75`, and a random seed of `54321`, create two line charts that demonstrate the performance on the **training** and **validation** sets during training: one line chart for a learning rate of `0.01` and another for a learning rate of `0.00001`.  The x-axis should be the epoch, the y-axis should be the loss and MAE, and there should be four lines -- one each for the `[loss, val_loss, mean_absolute_error, val_mean_absolute_error]` tracked by the `CSVLogger` during training.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `seoulbike_0.01.png`).

done

c. How did the accuracy curves of the two learning rates differ for the `penugins.csv` classification task?  How does this compare to the results you observed in Question 1?

We see huge increases in performance in the first 50 or so epochs on learning rate 0.01, and a more steady improvement the whole way to 250 on 0.00001. In line with results from question 1, where classification tasks seemed to prefer either 0.1 or 0.01. In the same vein as responses to q3, it appears that when learning rate is properly conifgured, the accuracy curves improve more dramitically. 

d. How did the loss curves of the two learing rates differ?  Were there any common trends across the data sets? Did any level off to a near-constant value?  Were any continually decreasing?  

Across both datasets and both learning rates, two things held true. First, both models prefer 0.01 to 0.00001 on these datasets. Second, when learning rate is properly configured, we should see large changes in validation loss and training loss within the first 50 epochs. If the learning rate is too slow like in this example, we expect to see steady decreases in loss all the way to 250 epochs. Both datasets on 0.01 appear to make most of their progress before 50 epochs before setlling at a low constant rate of improvement. On the contrary small learning rates like 0.00001 appear to induce continually decreasing loss curves. 

e. What do the shapes of the loss curves for the two learning rates imply?  Is there a relationship between the number of epochs needed for training a neural network model and the learning rate used?

These loss curves imply that you need more epochs at smaller learning rates. Which seems to track logically as well knowing that we make less progress at every epoch on smaller learning rates. Therfore it would take longer to converge. This relationship implies that we should allow models that seem to prefer smaller learning rates (closer to 0) more epochs to train. 

# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?

This was a very satifying assingment to complete before starting the questions. Seeing the validation accurary and then the training accuracy steadily increase when everything was properly configured was very satisfying to me. I enjoyed most programming the neural networks. I found the questions and plotting to be the most difficult across the board, and I wish I had some R-skills to lean on here. 

2. Approximately how much time did you spend on this assignment?

This took me the longest out of any homework assingment to date and took me more than 15 hours working alone. I likely spent around 2 hours just running the simulations for the first several line graph questions. I am sure this would be much more managable with a partener, but I found it took me quite a long time to create and debug the program before even starting the questions. 

3. Did you adhere to the Honor Code?

I have adhered to the honor code in this assessment.
Will Kennedy
