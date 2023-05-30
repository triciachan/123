# -*- coding: utf-8 -*-
"""Week_8_bnb_Query.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ri7eylOuRqrV0lMwqm25GLa2D8Stak0V
"""

# 1. required packages

import pandas as pd
import numpy as np

#2: Format to two decimals

pd.options.display.float_format = "{:,.2f}".format

#df.to_csv("BNB.csv",index=False)

# Load CSV data, 讀取資料
df = pd.read_csv("https://raw.githubusercontent.com/cchuang2009/2022-1/main/Python_IM/2023-2/BNB.csv")

df.head()

df[['price','room_type','host_since','zipcode','number_of_reviews']].nunique()

df[['price','room_type','host_since','zipcode','number_of_reviews']]

# Drop the duplicates by adding a new column “host_id”
df['host_id'] = df['price'].map(str)+df['room_type'].map(str)+ df['host_since'].map(str)+df['zipcode'].map(str)+df['number_of_reviews'].map(str)

df1 = df[['host_id','number_of_reviews','price']].drop_duplicates()
df1.info()


#Drop the duplicates without creating a new column.
df2 = df[['number_of_reviews', 'price']].drop_duplicates()

df2.info()

df['host_id']= df['price'].map(str)+df['room_type'].map(str)+df['host_since'].map(str)+df['zipcode'].map(str)+ df['number_of_reviews'].map(str)
df1 = df[['host_id','number_of_reviews','price']].drop_duplicates()

df1

# Commented out IPython magic to ensure Python compatibility.
# Visualization

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# %matplotlib inline

# distplot of all
sns.displot(df1,x='number_of_reviews', bins=10);
# distplot of majority
#sns.displot(df1[df1['number_of_reviews']<50],x='number_of_reviews', bins=10);

# Conditional statements with the lambda function

df1['host_popularity'] = df1['number_of_reviews'].apply(lambda x:'New' if x<1 else 'Rising' if x<=5 else 'Trending Up' if x<=15 else 'Popular' if x<=40 else 'Hot')

# Draw a nested boxplot to show bills by day and time
sns.boxplot(data=df1,x="host_popularity", y="price")
sns.despine(offset=10, trim=True)

result= df1.groupby('host_popularity').agg(min_price=('price',min),avg_price=('price',np.mean),max_price = ('price',max)).reset_index()

result

