# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:41:40 2024

@author: Solomon
"""

import streamlit as st
import os 
import numpy as np
import pandas as pd
import collections
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fancyimpute import IterativeImputer # for imputation(e.g MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso # for penalised regression and feature engineering
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from functools import partial
import random

@st.cache_data # Saves data into a cache facilitates re-run
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)
 
st.sidebar.title("Presentaion Layout")

pages = ["Project Context", "Data Exploration", "Data Preprocessing","Feature Engineering", " Modeling","Conclusion"]

page = st.sidebar.radio("Go to", pages)

if page == pages[0] :
    
    st.write("### Project Context")
    
    st.write("This project is aimed at analysing the data of the London Fire Brigade in an attempt to create a model that will be capable of predicting the Attendance time.The attendance time is the time from when the first call was received at a firefighting unit till when the first machine(unit) arrives the scene of the incident. ")
    
    st.write("The data sets is divided in to two main groups, the mobilisation data set and the incident data set. The mobilisation data set was further divdided in to three different excel files, while the incident data set was further divided in to two excel files.")
    
    st.write("I first of all loaded and merged the data sets files in to two main data frames, the mobilisation data('full_mobi_df'),and the incident data('full_incidence_df'). Then i analysed the data taking care of duplicates. There were also cases of datatypes inconsistencies with regards to a particular feature common to both data sets.This was handled before merging them into one data frame(merged_data),handling missing values, performing imputation and modellings , prediction conclusion and recomendation.")
    
    st.image("skynews-london-fire_5832608.jpg")
    
elif page == pages[1] :
    st.write("### Data Exploration")
    
    st.write("#### Difficulties encountered")
    st.write("The size of the data was a major problem to me because of my limited computational resources.I tried imputation techniques such as the KNN waand it failed due to limited resources(Memory error) . After displaying the tables of both the incident and mobilisation data frames, i noticed the intersections of three columns: the 'IncidentNumber', 'CalYear' and 'HourOfCall'. The IncidentNumber being the unique identifier of cases was later used for merging both data frames. ")
    st.write("##### Displaying the merged mobilisation dataframe.")
    # File uploader
    uploaded_file = st.file_uploader("full_mobi_df.csv", type=['csv', 'xlsx'])
    df = pd.read_csv("full_mobi_df.csv")
    st.write("DataFrame:")
    st.write(df)
    st.write("One can clearly observe that the same incidence number might correspond to responses from two different stations, with differenzt mobilisation statistics.")
    st.write("I will priotise the incidence number whose data correspond to the smallest attendance time")
    st.write("The mobilization data consist of 2,342,274 rows and 22 features")
    st.write("Out of the 2,342, 274 rows of the mobilisation data set, 786,576 rows are duplicates which is aproximately 34% of the data.")
    st.write("The duplicated IncidentNumber with the smallest AttendanceTimeSecond value was retained and the others droped.")
    st.write("The full mobilisation data set without duplicates is now having 1,555,698 rows and  22 columns")
    st.write("##### Displaying the merged incident dataframe.")
    # File uploader
    uploaded_file = st.file_uploader("full_incidence_df.csv", type=['csv', 'xlsx'])
    df = pd.read_csv("full_incidence_df.csv")
    st.write("DataFrame:")
    st.write(df)
    st.write("The incidence data consists of 1,680,826 rows and 39 features. There are also no duplicates in this data set.")

    st.write("##### Merging mobilisation data and incident data by IncidentNumber column.")
    st.write("The two data frames were then merged for further analyses")
    # File uploader
    uploaded_file = st.file_uploader("merged_data.csv", type=['csv', 'xlsx'])
    df = pd.read_csv("merged_data.csv")
    st.write("DataFrame:")
    st.write(df)
    st.write("Before the merging i tried to find out the number of shared IncidentNumbercolums in both data frames. Supprisingly i noticed that there were only __1,046,976__ shared rows. Since the incidence data set hat no duplicates, i expected a total coverage of the mobilisation data set.")
    
    
    st.write("#### Preprocessing")
    st.write("The IncidentNumber row with the smallest AttendanceTimeSeconds was retained.")
    st.write("The Calyear being  numerical, starting from 2009 was readjusted such that __2009==1__ so as to prevent subsiquent missinterpretation by algorithms ")
    st.write("#### Assessment of missing Values")
    uploaded_file = st.file_uploader("percent_missing_df.csv", type=['csv', 'xlsx'])
    df = pd.read_csv("percent_missing_df.csv")
    st.write("DataFrame:")
    st.write(df)
    st.write("Display of Percentage of missing values.")
    
    st.write("#### Barplot of Features and their Percentage of Missing Values")

# Display the saved barplot
    st.image('barplot_missing_values.png', caption='Plot of Features and their Percentage of Missing Values')
    st.write("It is so far unclear how features with high rate of missing values can be helful in calculating the outcome of interest. ")
    st.write("__Let us go further and analyse the data__")
# Page 2: Datanalysis
elif page == pages[2]:
    st.write("### Data Preprocessing")
    st.write("#### Handling Missing Values")
    st.write("Three types of missing values exist: __MCAR__,__MAR__,__MNAR__ and knowing the kind of missing values enables you decide the kind of imputaton will best handle them.")
    st.write("Imputation is a method used to fill in missing data with plausible values. ")
    st.write("The goal of imputation is to minimize the bias and maintain the integrity of the dataset, allowing for robust statistical analysis.")
    st.write("However imputation is best practised when the percentage of missing data is not greater than __5%__ and also if the data is numeric")
    st.write(" For this reason i am excluding all features with more than 10% of missing values")
    st.write("Next features with no missing values were also excluded before performing imputation. Below is a table of the tobe imputed featues and their respective percentages of missing values.")
    uploaded_file = st.file_uploader("with_missing_values", type=['csv', 'xlsx'])
    df = pd.read_csv("with_missing_values.csv")
    st.write("DataFrame: With __0<=missing_values<=10__")
    st.write(df)
    
    
    
    st.write("The above table contain both numeric and categorical features.Categorical data that have missing values will not be subjected to imputation. ")
    st.write("I extracted the numerical features from the set of features with missing values for imputation.")
    st.write("I performed imputation on the missing data using a variety of techniques including mean, median, MICE and KNN imputation but the KNN faild due to memory problem")
   
    st.write("#### Imputation by taking the mean")
    st.write("This method replaces the missing values with their mean.")
    st.write("This method can lead into severely biased estimates __even if data are MCAR__ on  statistical measures such as variance and covariance.")
    st.write("By comparing the results after with those before imputation, i also saw that the mean imputation has also worked well with both having very similar values __(243.5721053097065 vs 243.57210530968894 )__")
    st.write("I also perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors")
    
    st.write("#### Imputation by taking the median")
    st.write("Similar to mean, median imputation replaces missing values with the median of the observed data.")
    st.write("By using median istead of mean , the imputation process is less influence ba outliers nevertheless there is still some lost in the natural variability in the data especially when the data is not MCAR")
    st.write("The median is often a better representative of central tendency in skew distributions.")
    st.write("I also did Imputation by taking the median, then perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors just i was done for the other methods above. ")
    
    st.write("#### Mice Imputatation: Multiple imputation by chained equations")
    st.write("MICE imputation takes into account the variation and covariance between variables, which helps preserve the natural variability in the data.")
    st.write("MICE uses information from multiple variables simultaneously to estimate missing values, resulting in more accurate and robust estimates")
    st.write("MICE can handle complex patterns of missing data, including __MCAR (Missing Completely At Random), MAR (Missing At Random), and MNAR (Missing Not At Random).__")
    st.write("After performing __MICE imputation__, i compared the mean of the imputed data with the unimputed data using the __TravelTimeSeconds__ and found the difference to be very small __(243.52911704834688 vs 243.57210530968894)__. This implies the imputer performs well")
    st.write("There also existed a strong correlation between between the __TravelTimeSeconds and AttendanceTimeSeconds__ corr = 0.9285748213761903 ")
    st.write("I also Perform a linear regression to estimate R-squared value when we regress the outcome of interest and imputed predictors")
    
    st.write("#### Check which imputation approach is better")
    st.write("The fact that median imputation is slightly better than mean imputation indicates that the distribution is skewed.")
    uploaded_file = st.file_uploader("Imputation Methods", type=['csv', 'xlsx'])
    df = pd.read_csv("Imputation Methods.csv")
    st.write("DataFrame: Evaluation of/comparism of the imputation methods")
    st.write(df)
    st.write("The table above shows that MICE imputation leads to predictors that best explain the outcome of interest. This implies that a linear regression constructed from MICE-imputed values, almost perfectöy explain 100% of the variation in the model.")
    st.image("imputation_comparison.png")
    st.write("###### Verifying the Correlation between features")
    st.title("Correlation Heatmap")
    st.image('correlation_heatmap.png', caption='Pearson correlation between MICE imputed variables in the dataset')
    st.write("The Heatmap above shows us the correlation without values. A table of values can be found below.")
    uploaded_file = st.file_uploader("Correlation between features", type=['csv', 'xlsx'])
    df = pd.read_csv("Correlation between features.csv")
    st.write(df)
    st.write("From the plot above, we can already suspect autocorrelation between predictor variables and this can affect the performance of the model.")
    st.write("It can be seen that certain values best correlate with the outcome of interest.")
    st.write("Finally i return the imputed missing values in to the original dataframe")
    uploaded_file = st.file_uploader("merged_data2", type=['csv', 'xlsx'])
    df = pd.read_csv("merged_data2.csv")
    st.write("Dataframe with imputed missing values of numerical columns: shape = 1046976 X 14 ")
    st.write(df)
elif page == pages[3]:
    st.write("#### Feature engineering")
    st.write("I want to identify using __lasso regression, random forest regression and mutal information regression__ the features that are best predictive of the outcome of interest which is the AttendanceTimeSeconds.")
    st.write("The data was divided into the train and test data sets in the ratio 7:3 respectively so as to be able to perform cross validation.")
    st.write("The dimension of X_train is (732883, 13) and The dimension of X_test is (314093, 13)")
    st.write("Standardisation of the features was made using StandardScaler")
    st.write("#### Lasso Regression")
    st.write("__Lasso regression__ using the LassoCV on the training and testing data sets. The following results were obtained;")
    st.write("The __train__ score for ls model is 0.9999977945446924 and The __test__ score for ls model is 0.9999978341637061")
    st.image('lasso_coefficients_plot.png', caption='Lasso Regression Coefficients')
    st.write("It excludes all features that have very low or insignificant predictive power. It only identiy three features that are predictive of the outcome. ")
    st.write("##### Random forest regression")
    st.write("I perform an analysis of feature importance using random forest regression.")
    st.write("The plot below show a review of feutures importance by random forest regression begining from the most important to the least important.")
    st.image("random_forest_feature_importances.png")
    st.write("#### Mutual information regression")
    st.write("This helps us identify top k features that are best predictive of the outcome, using mutual information. This approach can pick up both linear and non-linear relationships.")
    st.write("Below is a bar plot of the 10 best predictive features from MIR")
    st.image("mutual_information_regression_scores1.png")
    st.write("Below is a correlatio plot of the 10 best features selected by Mutual information regression")
    st.write('#### MIR_Features CorrelationHeatmap')
    st.image('MIR top 10 corr_heatmap.png')
    # Add content for Data Analysis

# Page 3: Modeling
elif page == pages[4]:
    st.write("### Modeling")
    st.write("The first model that i tried  is the linear model(lm)")
    st.write("Linear modeling( First Mode)")
    st.write("The data was splitted in to the train and test sets in the ratio 7:3")
    st.write(" For the pridictors/independent variables, the shape of the train data:__(732883, 10)__ and \nshape of test data: __(314093, 10)__ \n  ")
    st.write("For the target/dependent variable,the shape of the train data:__(732883)__ and \nshape of test data: __(314093)__ \n  ")
    st.write("The predictors were standardised and then followed by the creation of the model")
    st.write("After the model's creation, the model was tested by plotting actual vavlues of __AttendanceTimeSeconds__ from the test data set against the predicted values of __AttendanceTimeSeconds__ from the trained model. Below is a diagram of the plot.")
    st.image("r_sqr_lm.png")
    st.write("The R-sqr score of this lm is 1.0 meaning that the model has learned the relationship between the features and the target variable so well that it can predict the target variable for the training instances with 100% accuracy")
    st.write("The train score is 1 : meaning that the model has learned the relationship between the features and the target variable so well that it can predict the target variable for the training instances with 100% accuracy.")
    st.write("The test score is 0.9999865816025735 This suggests that the model's predictions are very close to the actual values.")
    st.write("Although R_sqr results eveluates how well the model can predict, they are at time misleading thus, should not be use in isolation.")
    st.write("#### Below is a plot of Residuals Vs fitted")
    st.image("residual_analysis.png")
    st.write("I observe unsual patterns in this plot. I did not expect to see any visible trend in this didtibution residuals vs fitted")
    st.write("This could be because many of the predictors had redundant values among the observation e.g HourOfCall")
    st.write("I performed linear regression again for hypothesis testing, to see how each predictor contributes to the attendance time")
    st.title('OLS Regression Results')
    with open('regression_summary.txt', 'r') as f:
        regression_summary = f.read()
    st.text(regression_summary)
    st.write("The RMSE for the linear model is 0.5201456329770627. To better interprete this value, i will compare it the value obtained from random forest model trained on the same data set. ")
    st.write("#### Second Model")
    st.write("##### Random forest regression with optimal parameters")
    st.write("In this second model(RF), the calculated RMSE is 6.056569838315212 ")
    st.write("By comparing these values, i see that the linear model outperforms the random forest regression model.")
    st.write("#### Selection of significant features from hypothesis testing")
    st.write("The first hypothesis testing above gave us an overview of some statistically significant features")
    st.write("I selected from the linear model features with p-values less than 5%. These are features that are statistically significant")
    st.write("This gave me a final list of features cantining 4 features:__CalYear', 'FirstPumpArrivingAttendanceTime', 'TurnoutTimeSeconds' and 'TravelTimeSeconds'__ ")    
    st.write("Refitting the linear model only using significant features") 
    # Linear model only with significant features.
    #x_lm = sm.add_constant(X[final_features])
    #lm_final = sm.OLS(y, x_lm).fit()
    #y_pred = lm_final.predict()
    st.write("A plot of actual Vs fitted AttendanceTimeSec looks so much the same as the previous above.")
    st.image("actual_vs_fitted2.png")
    st.write("I moved on and evaluated  the residuals of the  optimal model.")
    st.write("Evaluation of the optimal linear model.")
    st.image("residual_analysis2.png")
    st.write("There is a visible reduction in the value of the RMSE for the optimised model with statiticall significant features")
    st.write("RMSE_first_lm =0.5201456329770627 , RMSE_optimized_lm = 2.5896974487841527e-09 ")
    st.write("Below is a correlation plot of the final features with our outcome of interest ; __AttendanceTimeSeconds__")
    st.image("corr_final_features.png")
    st.write("We can clearly see how the significant features are correlated with the target variable.")
    st.write("##### The created model for calculatin or predictin AttendanceTimeSecond")
    st.image("AttendanceTimeSeconda_Calculator.png")
    
    # Add content for Modeling

# Page 4: Conclusion
elif page == pages[5]:
    st.write("### Conclusion")
    st.write("I used several approaches to to identify features that best predict the attendance time.  ")
    st.write("These features are __CalYear, 'FirstPumpArriving_AttendanceTim, 'TurnoutTimeSecon and 'TravelTimeSecc.__")
    st.write("Linear regression shows that the two outstanding contributors are the __turnout time__ and even more importantly the __travel time__.")
    st.write("The travel time in turn has a correlation with the attendance time of the first arriving pump.")
    st.write("The travel time can hence be prioritised for action. This can be done by putting in place machanisms to enable LFB to deploy a team to the incidence zones on time")
    st.write("The Gonernment can consider investing in the creation of drones that can help in fire fighting.")
    st.write("##### Thanks for your attention")
    st.write("Auf Wiedersehen")
    st.write("Au revoir")
    st.image("Goodbye.jpg")
    # Add content for Conclusion