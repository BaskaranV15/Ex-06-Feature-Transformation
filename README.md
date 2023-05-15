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
![Screenshot 2023-05-15 114611](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/6b40fe2b-df4c-4c8d-a4eb-ac15073aa87c)
![Screenshot 2023-05-15 114623](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/fa1e8147-fa3b-40e4-b344-7ecccb1d755c)
![Screenshot 2023-05-15 114634](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/aae66c1e-0b09-4453-9c82-d37d10ce39f9)
![Screenshot 2023-05-15 114642](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/4703d29e-1b96-4af8-b9ae-ae0960353018)
![Screenshot 2023-05-15 114655](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/27217e62-62ac-49a2-9182-76aacf3f265c)
![Screenshot 2023-05-15 114705](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/4963a43b-50e2-4445-bde9-f5f791857fbf)
![Screenshot 2023-05-15 114713](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/657369fe-6f41-4e4b-bd09-c47073c6e442)
![Screenshot 2023-05-15 114721](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/4f4e6050-4e46-4e71-b3bd-52de8c1d3143)
![Screenshot 2023-05-15 114729](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/178ea12d-a116-48ec-abe2-64b713ea6afd)
![Screenshot 2023-05-15 114737](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/837af8f8-4abd-4d9e-a531-0cd3657d65f5)
![Screenshot 2023-05-15 114756](https://github.com/BaskaranV15/Ex-06-Feature-Transformation/assets/118703522/c649b121-fd02-40d7-bda7-45e72cb6d2f1)


