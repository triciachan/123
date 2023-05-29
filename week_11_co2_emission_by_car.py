# -*- coding: utf-8 -*-
"""Week_11_CO2_Emission_by_CAR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19WO4v7xnA-EhzeCGo7SU4DtYqLG3ZbIn

# ML Modeling, Linear Model
Goal
---
1. scikit-learning Introduction
2. CO2 emisssion 

Reference
---
1. [src](https://medium.com/joguei-os-dados/week-3-predicting-co2-emissions-70e554ad2276)
2. [Dataset, Kaggle](https://www.kaggle.com/gangliu/oc2emission/tasks), dataset contains information of CO2 emission from cars.

Questions
---
1. what is the correlation between the engine motor of the car and its CO2 emission? 
2. Is it possible to create a model that predicts the amount of polution on future motors? How and Why?
"""

# importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

#matplotlib inline

## atributing dataset to a dataframe df
df = pd.read_csv('/content/FuelConsumptionCo2.csv')
df.head(2)

# describing statiscal measures
df.describe()

"""### Exploratory analysis
1. correwlation between pairs
"""

df.info()

num_features=list(df.select_dtypes(include='number').columns)

# correlation matrix to measure the strenght of the correlation between features
plt.figure(figsize=(13,5))
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# correlation matrix to measure the strenght of the correlation between features
plt.figure(figsize=(13,5))
corr = df[num_features[1:]].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# only one category
df['MODELYEAR'].nunique()

# ploting graph engine size x CO2 emissions
plt.figure(figsize=(13,5))
sns.lineplot(x=df['ENGINESIZE'], y=df['CO2EMISSIONS'])
plt.xlabel('Motor engine')
plt.ylabel('CO2 emissions')
plt.show()

"""The lineplot shows a positive correlation between the size/power of the engine motor and the carbon emission. With some variation, we can say that the bigger the engine the greater the levels of CO2 emited.

[BONUS]
---
What are the differences of the techniques of Car manufacture among different countries?
"""

!pip install mplcyberpunk

df['MAKE'].unique()

df.groupby('MAKE')[['ENGINESIZE','CYLINDERS']].agg(pd.Series.mode)
#com_df.loc['TOYOTA','ENGINESIZE']=2.5
#com_df.loc['BENTLEY','CYLINERS']=10

com_df=df.groupby('MAKE')[['ENGINESIZE','CYLINDERS']].agg(lambda x:x.mode()[0])#.reset_index()

com_df.index



com_df=com_df.loc[['AUDI','BMW','CHEVROLET','HYUNDAI', 'LINCOLN','PORSCHE','TOYOTA']]
com_df

MAKER=list(com_df.index)
Enginesize=list(com_df['ENGINESIZE'])
Cylinders=list(com_df['CYLINDERS'])

MAKER = [*MAKER, MAKER[0]]
Enginesize = [*Enginesize, Enginesize[0]]
Cylinders = [*Cylinders, Cylinders[0]]

Maker = np.linspace(start=0, stop=2 * np.pi, num=len(MAKER))

MAKER

Enginesize

from matplotlib.patches import Patch
import mplcyberpunk

with plt.style.context('cyberpunk'):
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

    ax.plot(Maker, Enginesize, lw=2)
    ax.plot(Maker, Cylinders, lw=2)

    ax.fill(Maker, Enginesize, alpha=0.3)
    ax.fill(Maker, Cylinders, alpha=0.3)

    lines, labels = plt.thetagrids(np.degrees(Maker), labels=MAKER)

    ax.tick_params(axis='both', which='major', pad=30, labelsize=15)

    ax.spines['polar'].set_linewidth(3)
    
    edge_color = (1, 1, 1, 0.2) 
    ax.spines['polar'].set_color(edge_color) 
    
    ax.grid(color='white', alpha=0.3)
    
    ax.set_ylim(0, 10)
    
    # Create custom legend handles
    Enginesize_legend = Patch(facecolor='C0', alpha=0.5, label='Enginesize')
    Cylinders_legend = Patch(facecolor='C1', alpha=0.5, label='Cylinders')

    # Add a legend with custom position and handles
    ax.legend(handles=[Enginesize_legend, Cylinders_legend],
              bbox_to_anchor=(1.2, 1.2), fontsize=20, 
              frameon=True)


    plt.show()

"""Model selection
---
1. what is the suitable model?
2. ML Modeling 
3  prediction/Estimation and Error discussion

### Spliting data to train the model
"""

# importing necessary libraries
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

# features into variables
engine= df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]

# ploting the correlation between features
plt.scatter(engine, co2, color='blue')
plt.xlabel('engine')
plt.ylabel('co2 emission')
plt.show()

"""The datapoints on the scatterplot indicates that is possible to do a linear regression with this dataset, although the amount of residual."""

# spliting data in train and test with train_test_split
engine_train, engine_test, co2_train, co2_test = train_test_split(engine, co2, test_size=0.2, random_state=42)

# ploting the correlation between features
plt.scatter(engine_train, co2_train, color='blue')
plt.xlabel('engine')
plt.ylabel('co2 emission')
plt.show()

"""### Creating the model with the train dataset"""

# creating a linear regression model
# LinearRegression is a method of sklearn
modelo = linear_model.LinearRegression()

# linear regression formula: (Y = A + B.X)
# training the model to obtain the values of A and B (always do it in the train dataset)
modelo.fit(engine_train, co2_train)

"""Estimated Linear Model
---
Data: $(x^i,y^i)$'s pair
$$ \hat y= A x+B$$
"""

# exibiting the coeficients A and B that the model generated
print(f'(B) intercept: {modelo.intercept_} | (A) inclination: {modelo.coef_}')

# print linear regression line on our TRAIN dataset
plt.scatter(engine_train, co2_train, color='blue')
plt.plot(engine_train, modelo.coef_[0][0]*engine_train + modelo.intercept_[0], '-r') 

plt.ylabel('CO2 emissions')
plt.xlabel('Engine')
plt.show()

"""### Executing the model on the test dataset
First: predictions on the 'test' dataset
"""

predictCO2 = modelo.predict(engine_test)

# print linear regression line on our TEST dataset
plt.scatter(engine_test, co2_test, color='green')
plt.plot(engine_test, modelo.coef_[0][0]*engine_test + modelo.intercept_[0], '-r')
plt.ylabel('CO2 emissions')
plt.xlabel('Engine')
plt.show()

"""### Evaluating the model"""

# Showing metrics to check the acuracy of our model
print(f'Sum of squared error (SSE): {np.sum((predictCO2 - co2_test)**2)}') # SSE: sum all of the  residuals and square them. 
print(f'Mean squared error (MSE): {mean_squared_error(co2_test, predictCO2)}') # MSE: avg of SSE
print(f'Mean absolute error (MAE): {mean_absolute_error(co2_test, predictCO2)}')
print (f'Sqrt of mean squared error (RMSE):  {sqrt(mean_squared_error(co2_test, predictCO2))}') # RMSE: sqrt of the MSE
print(f'R2-score: {r2_score(predictCO2, co2_test)}') # r2-score: explains the variance of the variable Y when it comes to X

"""All of the metrics above help evaluate the acuracy of the model. r2, for instance, is 0.68: this means that our linear regression model (values A and B given) is able to explain 68% of the variance between the CO2 emission and engine of the cars. 

The usual benchmark for this metric is 0.70.

[Group Practicing]
---
As same as the processing for estimated between `Engingesize ➡︎ CO2 emission`, now consider make a practicing for
`Cylinders ➡︎ CO2 emission`.
"""

# features into variables
cylinders= df[['CYLINDERS']]
co2 = df[['CO2EMISSIONS']]

def LinearModel(X,y,features_,split_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
    reg_model = linear_model.LinearRegression()
    reg_model.fit(X_train, y_train)
    # exibiting the coeficients A and B that the model generated
    #print(f'(B) intercept: {reg_model.intercept_} | (A) inclination: {reg_model.coef_}') 
    #print("Estimated Linear Model\n [CO2]=")
    # Get the coefficients and intercept
    coeffs = reg_model.coef_
    intercept = reg_model.intercept_

    # Create a string representation of the expression
    expr = "[CO2 Emission] = {:.2f}".format(intercept)
    for i in range(len(coeffs)):
        expr += " + {:.2f}[{}]".format(coeffs[i], features_[i])

    # Print out the expression
    print("===\n",expr,"\n===")
    
    predict_= reg_model.predict(X_test)
    if (len(X)==1):
        plt.scatter(X_test, y_test, color='green')
        plt.plot(X_test, reg_model.coef_[0][0]*X_test + reg_model.intercept_[0], '-r')
        plt.ylabel('CO2 emissions')
        plt.xlabel(feature_)
        plt.show()
    
    
    # Showing metrics to check the acuracy of our model
    print(f'Sum of squared error (SSE): {np.sum((predict_ - y_test)**2)}') # SSE: sum all of the  residuals and square them. 
    print(f'Mean squared error (MSE): {mean_squared_error(y_test, predict_)}') # MSE: avg of SSE
    print(f'Mean absolute error (MAE): {mean_absolute_error(y_test, predict_)}')
    print (f'Sqrt of mean squared error (RMSE):  {sqrt(mean_squared_error(y_test, predict_))}') # RMSE: sqrt of the MSE
    print(f'R2-score: {r2_score(predict_, y_test)}') # r2-score: explains the variance of the variable Y when it comes to X

X=df[['CYLINDERS']]
y=df['CO2EMISSIONS']
features=list(X.columns)

LinearModel(X,y,features)

X=df[['ENGINESIZE']]
y=df['CO2EMISSIONS']
features=list(X.columns)

LinearModel(X,y,features)

X=df[['ENGINESIZE','CYLINDERS']]
y=df['CO2EMISSIONS']
features=features=list(X.columns)
LinearModel(X,y,features)

from mpl_toolkits.mplot3d import Axes3D

# Fit the linear model
model = linear_model.LinearRegression().fit(X, y)

# Get the coefficients and intercept
A1, A2 = model.coef_
C = model.intercept_

# Create a meshgrid of X and Y values
X1, X2 = np.meshgrid(X['ENGINESIZE'],X['CYLINDERS'])
Y = A1*X1 + A2*X2 + C

# Create the 3D plot
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
#ax.scatter(X['ENGINESIZE'], X['CYLINDERS'], y)

# Plot the estimated plane
ax.plot_surface(X1, X2, Y, alpha=0.01)

# Add labels and title
ax.set_xlabel('ENGINESIZE')
ax.set_ylabel('CYLINDERS')
ax.set_zlabel('CO2')
ax.set_title('Estimated Linear Model')

# Show the plot
plt.show()

import plotly.graph_objects as go

# Create the 3D plot
fig = go.Figure()

# Add the data points to the plot
fig.add_trace(go.Scatter3d(x=X['ENGINESIZE'], y=X['CYLINDERS'], z=y, mode='markers', marker=dict(color='blue')))

# Add the estimated plane to the plot
fig.add_trace(go.Surface(x=X1, y=X2, z=Y, opacity=0.5, colorscale='Viridis'))

# Update the layout of the plot
fig.update_layout(scene=dict(xaxis_title='ENGINESIZE', yaxis_title='CYLINDERS', zaxis_title='CO2'))

# Show the plot
fig.show()

"""[Pycaret](http://pycaret.org), Low-code Package
---
Less operation, more result. What its goal is to provide handy way to make ML model. However it is built on plenty of packages and this makes a little trouble in installation. 

Install in Google colab
---
```
!pip install -U xgboost catboost category_encoders --ignore-installed --no-deps
!pip install --pre --no-deps pycaret
!pip install -U sktime  scikit-plot scikit-learn
```
Whatever package is absent, use `pip` to install.
"""

from pycaret.regression import *

data=df[['ENGINESIZE','CYLINDERS','CO2EMISSIONS']]
s = setup(data = data, target = 'CO2EMISSIONS', session_id=123)

best = compare_models()

catboost = create_model('catboost')

tuned_cat = tune_model(catboost)

predict_model(tuned_cat);

final_cat = finalize_model(tuned_cat)
print(final_cat)

save_model(final_cat,'Final_cat_2023_4')

saved_final_cat = load_model('Final_cat_2023_4')

print(saved_final_cat)

predict_model(saved_final_cat)

"""BY chatGPT
```
SHAP, SHapley Additive exPlanations

In a SHAP plot, each feature is represented by a vertical bar. The horizontal position of the bar represents the SHAP value of that feature. The color of the bar represents the value of the feature for that instance, where red indicates a high value and blue indicates a low value. The length of the bar indicates the impact of the feature on the output. If the bar extends to the right of the zero point, it indicates a positive impact, and if it extends to the left, it indicates a negative impact.
```

"""

# interpret summary model
interpret_model(saved_final_cat, plot = 'summary')

create_app(saved_final_cat)

!pip install -U gradio uvicorn fastapi

interpret_model?

