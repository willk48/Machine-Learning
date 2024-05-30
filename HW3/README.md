[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/l1FvqcXt)
# HW3: Regression with Weighted Models
Name: Will Kennedy

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW3/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW3/)

# Data Sets 

This assignment contains five data sets that are based on publicly available benchmarks:

1.	**capitalbike.csv**: A data set describing bike rentals within the Capital bikeshare system.  The task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 12 attributes are a mix of 6 categorical and 6 numeric attributes, including information such as the season, day of the week, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

2.	**seoulbike.csv**: Another data set describing bike rentals in a metropolitan area (Seoul, South Korea).  Again, the task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 11 attributes are a mix of 2 categorical and 9 numeric attributes, including information such as the season, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

3.	**energy.csv**: A data set describing the energy consumption in 10-minute increments by appliances in a low-energy residence in Belgium.  The task is to predict how much energy was consumed by appliances.  Each of the 27 attributes are numeric and describe measurements from sensors in the residence or nearby weather stations, as well as energy usage by lights.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

4.	**forestfires.csv**: A data set describing forest fires in northeastern Portugal.  The task is to predict how much area was burned by forest fires.  The 12 attributes are a mix of 2 categorical and 10 numeric values, including date and weather data, as well as the geographic location of the area within the Montesinho park.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/162/forest+fires](https://archive.ics.uci.edu/dataset/162/forest+fires)

5.	**wine.csv**: A data set of measurements of wine samples.  The task is to predict the quality of the wine (on a numeric scale).  The attributes are a mix of 11 numeric measurements from the wine, along with 1 categorical attribute describing the color of the type of wine.  This data set is the most popular regression task from the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)


# Research Questions

Please answer the following research questions using the `regression.py` program that you created.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices and a value of `true` for whether you want to rescale the numeric attributes using max-min normalization, train the requested eight models on each of the five data sets and calculate the MAEs on the corresponding test sets.  (You do *not* have to list those MAEs here)

Seed: 54621
Training Percentage: 75

a. Using the resulting MAEs, create one **bar chart** for each data set and save the five bar charts to image files that clearly describe to which data set they correspond (e.g., `capitalbike_rescaled_bar.png`).  Upload those five image files to your GitHub repository.

Done

b. Comparing the performance of Linear Regression and Ridge Regression across the five data sets, what trends do you observe?  Does adding regularization improve the performance of Ridge Regression over Linear Regression?  Why do you think this result occurred?

There were very marginal differences between lin_reg and ridge in these 5 data sets. In sets Capitalbike, Seoulbike, and Wine there was almost no difference between the two. In datasets Forestfires and Energy, ridge had a distinguishable advantage over lin_reg. I think this is becuase those two datasets must have a higher incidence of independent variables that effect the result. Energy and forest fires are likely to be overfit: forest fires becuase it is so small and energy because of the high attribute count. In such situations ridge should have the edge.

c. Comparing the performance of Linear Regression and LASSO across the five data sets, what trends do you observe?  Does adding regularization improve the performance of LASSO over Linear Regression?  Why do you think this result occurred?

Lasso performed equal or worse when compared to lin_reg in 4 out of the 5 datasets surveyed and only had a defined edge in forestfires. Lasso appears to be useful in minimizing the influence of non-important variables. I think it showed decreased MAEs on the forestfires dataset again because the dataset was too small and thus minimizing the weights of non-essential variables had a higher influnece on the predictions made. In the case of why it performed the same or worse on the rest of the datasets, I would say that is because we were using it outside it intended utility of selecting variables. 

d. Comparing the performance of Linear Regression and the four SVM models across the five data sets, what trends do you observe?  Do the improvements of the SVM approach lead to improved predictions?

SVM models were shown in every dataset to outperform lin_reg. SVMs allow for non-2D decision boundaries and thus can transform data into higher dimensional spaces as compared to the linear regressors.  

e. Comparing the performance of the different kernels within the four SVM models across the five data sets, what trends do you observe?  Is one kernel a better choice for these data sets than the other?  How does the choice of degree for the Polynomial kernel affect learning performance?

The overall trends established by the SVM models were that higher kernel values led to increased performance in all test cases. The RBF had mixed results showing very minor improvements in 2/5 datsets, but demonstrated distinctly worse results in the 3 others.  

f. Comparing the performance of the Decision Tree with the other models across the five data sets, what trends do you observe?  When did the decision tree do better or worse than the other models?  Why do you think this might have occurred?

The decision tree outperformed the other models by a significant margin in both bike datasets, came in first by a slim margin on wine, and came in third on the energy dataset. The tree performed quite bad on the forestfires dataset closely resembling the linear regressors. These trends are likely due to the low degree of overfitting that happens in trees. If we can afford the time complexity of using a decision tree, decision trees should outperform the linear regressors everywhere the data is non-linear. Trees can better fit non-linear data by making only the most useful distinctions and omitting non-useful variables. 

#### Question 2

Using the same random seed and training set percentage as in Question 1 and a value of `false` for whether you want to rescale the numeric attributes using max-min normalization, rerun your program to train the requested eight models on each of the five data sets and calculate the resulting MAEs on the test sets.  (You do *not* have to list those MAEs here)

Seed: 54621
Training Percentage: 75

a.  Using the resulting MAEs, create one **bar chart** for each data set and save the five bar charts to image files that clearly describe to which data set they correspond (e.g., `capitalbike_unscaled_bar.png`).  Upload those five image files to your GitHub repository.

Done

b. Comparing the performances of Linear Regression within the five data sets, do you observe any changes in the performance of the model whether or not you rescale the numeric attributes?  Why do you think this result occurred?

We see little to no change in results when attributes are rescaled in lin_reg. Closed form solutions like sci-kit learns implemntation of lin_reg do not benefit often from feature scaling because it is a closed loop solution to lin_reg. 

c. Comparing the performances of LASSO and Ridge Regression within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

Similar to the last point, we see no significant change after scaling in most datasets. In 4/5 datasets we see no improvement with Ridge after scaling only showing a small increase in performance in the forest fires dataset. For LASSO, in 3/5 sets performance was the same before and after scaling, in forest fires it performed better after scaling, and performed worse on the energy dataset after scaling. I am unsure why exactly we see different trends on these 2 sets which both contain a lot of attributes. If I had to guess it would be because forestfires has the smallest dataset but may have a high range in the data that would see benefits from scaling. However, energy is the largest dataset and LASSO is sensitive to outliers so you would think it would get better. It likely got worse because of the sheer size of our largest dataset. 

d. Comparing the performances of SVM within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

SVM performed better in 4/5 datsets after scaling only showing similar results in forestfires. This is likely because SVMs perform a lot of distance calculation in higher dimensional spaces, and normalization would decrease the distance between all points before and after scaling. 

e. Comparing the performances of the Decision Tree within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

There was no difference before and ofter scaling for any datasets on the tree model. This is due to the fact that trees split on features not necessarily scales. A tree would have the same splits before and after scaling in most cases. Trees are not senstive to variance. 

f. If given a new regression task, for which types of models might you consider rescaling numeric attributes?

I would consider rescaling on all SVM models and possibly ridge depending on the datasets I planned on using it on. 

#### Question 3

Using the **energy.csv** data set, a random seed of `12345`, and a value of `true` for whether you want to use max-min normalization for the numeric attributes, calculate the MAE on the test set for each of the eight models using each of the following training percentages: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`. (You do *not* have to list those MAEs here)

a. Create a **single line chart** plotting the test set MAE for each of the eight models for each of the training percentages.  That is, in your line chart, the training percentage should be the x-axis, the MAE should be the y-axis, and there should be one line for each of the eight models.  Save the line chart to an image file with the name `energy_rescaled_line.png` and upload it to your GitHub repository.

Done, in folder Q3

b. What trends do you observe as the training percentages increase?  Which models improved?  Which models had consistent performance?  Did any models do *worse* with more training data, and if so, which ones?

All models generally improve as training percentage increases up to the 70/30 split. At which point only tree improved from 70 to 80% and all other models suffered. Lasso was a bit strange and had local minimums at training percentages 10:90 and 70:30. This lines up with the perceived significance of the 70:30 split, and it is noteworthy that LASSO did not deviate much across the whole surveyed data, so changes are all small. Tree seemed to improve dramatically with every increase in training data all the way up to 80:20. Linear regressors did not improve much over increases in training percent, while SVMs saw steady improvement up to 70:30.

c. What do these trends imply about how you should choose a type of regresssion model, based on the amount of data you have available?  If you have a small training set, which model would you choose and why?  If you have a large training set, which model would you choose and why?

From the results from the graph I would generally always select an SVM with a high kernel count unless I had a sufficiently large dataset at which point I would switch to a decision tree. Decision tree never actually passed SVM_4 on the energy dataset, but it was still trending downward which implies that with higher counts it may surpass SVM_4.

#### Question 4

Using the **seoulbike.csv** data set, a random seed of `12345`, and a value of `true` for whether you want to use max-min normalization for the numeric attributes, calculate the MAE on the test set for each of the eight models using each of the following training percentages: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`. (You do *not* have to list those MAEs here)

a. Create a **single line chart** plotting the test set MAE for each of the eight models for each of the training percentages.  That is, as in Question 3, the training percentage should be the x-axis of your line chart, the MAE should be the y-axis, and there should be one line for each of the eight models.  Save the line chart to an image file with the name `seoulbike_rescaled_line.png` and upload it to your GitHub repository.

b. What trends do you observe as the training percentage increases that are **similar** to the trends in Question 3 with the **energy.csv** data set?

The trends that carry over from our last dataset are: that SVMs show increases in performance as training percentage increases and linear regressors remain relatively constant. As opposed to energy, SVMs improved up to and including the 80:20 split where our energy dataset showed slight degradation at 80:20. As I will get into on the next section, our linear regressors in energy were trending more accurate as training split increased, but now are trending more erroneous as training splits increase. 

c. What trends do you observe as the training percentage increases that are **different** from the trends in Question 3 with the **energy.csv** data set?

Trends that are different include: tree performs abnormally on training split 30:70 and outpaces the expected rate of improvement before settling, and linear regressors get worse slightly. In energy our tree model saw steady improvements at eevery increase in training percentage. However, this time around we see slight improvements everywhere except at 30:70 which you can see on the graph does not follow the pattern. With that result omitted it trends similarly to the energy dataset, but with much a much lower range. Our linear regressors trended more accurate by a slim margin on the energy set, but on this one trended slighty more erroneous.  

d. Based on your answers to Question 4b and 4c above, what does this reveal about the process of designing a machine learning solution?  Are there some general rules of thumb that we can trust to help us make choices (e.g., on what model to use)?  How can we know whether those choices were the right ones to make for a particular data set?

General rules that have appeared here seem to converge with our previous 'around 70 percent' classroom norms. Both datasets seemed to generally handle larger training splits better than smaller ones, suggesting that larger datasets uniformly improve accuracy. In the same vein, SVMs with polynomial kernel functions at 3 or 4 and decision trees produce higher accuracy at the cost of resources than linear regressors of any type. Depending on the variance and size of your datset, it may be useful to implement a fast closed loop linear regressor because you simply do not have enough data to see the performance increases of SVM_4 or CART. Otherwise, these experiments suggest that a in most cases, with sufficient data, SVM_4 and CART are the clear winners on accuracy. 

# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?

I enjoyed most the first 2 research questions. When you spend a long time on an assignment like this programming to be left with pretty nebulous solutions, it can be pretty underwhelming. It is hard to just drop the tables after successfully implementing a ML model and see the fruits of your labor. I liked creating the figures for each of the questions as it really let me see the value in the parts of the assignment coming together to help create a clear picture on what each of these models do.

2. Approximately how much time did you spend on this assignment?

Approx. 9 hours total: 5 programming, 4 on research questions and plt functions

3. Did you adhere to the Honor Code?

I have adhered to the honor code in this assessment.
