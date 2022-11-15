# HEART FAILURE CLINICAL RECORDS PREDICTION
## Random Forest Classifier
## Introduction
According to the World Health Organization, cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Of these deaths, 85% are due to heart attack and stroke. Heart failure is a common event caused by cardiovascular diseases.
The early detection of people with cardiovascular diseases or who are at high cardiovascular risk due to the presence of one or more risk factors is paramount to reducing deaths arising from heart failures. As a result, predictive models become indispensable.

## Objective
The main objective of this project is to explore the Heart Failure Dataset, and to apply Random Forest Classifier models of machine learning on it. Create a Streamlit web app for whole architecture.


## The Dataset
The dataset is composed by 299 patients with heart failure collected in 2015. For every patient were collected key parameters of their clinical picture which theoretically and realistically correlated with their status.



The features/variables/columns in the datasets are the following: 
- Age <integer> that contains the age of each patient at the time of the heart failure.
- Anaemia <factor> binary value which reveals the absence (0) or the presence (1) of Anaemia.
-	Creatinine Phosphokinase - CPK <integer> that contains the level of the CPK enzyme in the blood (mcg/L)
-	Diabetes <factor> binary value which reveals the absence (0) or the presence (1) of Diabetes.
-	Ejection Fraction - EF<numeric> that contains the title of each movie including the year of the release.
-	High Blood Pressure - HBP<factor> binary value which reveals the absence (0) or the presence (1) of hypertension.
-	Platelets - P<integer> that count the number of platelets.
-	Serum Creatinine - SC <integer> that contains the level of Serum Creatinine in the blood (mg/dL).
-	Serum Sodium - SS <integer> that contains level of Serum Sodium in the blood (mEq/L).
-	Sex <factor> binary value which reveals the sex. 0 if female, 1 if male
-	Smoking <factor> binary value which reveals the nicotine addiction. 0 if absent, 1 if present
-	Time <integer> that represents the follow up period (days)
-	Death Event <factor> binary value which reveals if the patient deceased during the follow-up period 1 or not 0;

## Random Forest Classification:
Random forest algorithms have three main hyperparameters, which need to be set before training. These include node size, the number of trees, and the number of features sampled. From there, the random forest classifier can be used to solve for regression or classification problems.
The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample. Of that training sample, one-third of it is set aside as test data, known as the out-of-bag (oob) sample, which we’ll come back to later. Another instance of randomness is then injected through feature bagging, adding more diversity to the dataset and reducing the correlation among decision trees. Depending on the type of problem, the determination of the prediction will vary. For a regression task, the individual decision trees will be averaged, and for a classification task, a majority vote—i.e. the most frequent categorical variable—will yield the predicted class. Finally, the oob sample is then used for cross-validation, finalizing that prediction.


