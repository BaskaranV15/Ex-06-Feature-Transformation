# Ex-06-Feature-Transformation
# AIM
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
# ALGORITHM
## STEP 1
Read a given data
## STEP 2
Clean the Data Set using Data Cleaning Process
## STEP 3
Apply Feature Transformation techniques to all the features of the data set
## STEP 4
Save the data to the file
# CODE
```python
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.preprocessing as s
import scipy.stats as stats
import statsmodels.api as sm
df=pd.read_csv("/content/Data_to_Transform.csv")
df
df1=df.copy()
df1
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
a=df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
a
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
## OUTPUT
