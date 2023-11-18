import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Students_gamification_grades.csv")

df['Total_Attempts']= df.iloc[:, 10:16].sum(axis=1)
df['Improvement_Rate'] = df['Final_Exam'] - df['Practice_Exam']

#Snippets of code used to get some of the averages discussed
'''
p = df[(df['Total_Attempts'] == 0)]
avg_improv = p.iloc[:, 16].sum()/41
print(p)
print(avg_improv)

p = df[(df['Total_Attempts'] > 0)]
print(p)
avg_prac = p['Practice_Exam'].mean()
print(avg_prac)
'''

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))
# Plot X=Student_Id and Y=Total attempts for practice quizzes 1-6 on the, first subplot
sns.lineplot(x='Student_ID', y='Total_Attempts', data=df, ax=ax0)
# Plot X=Student_Id and Y=Improvement rate from practice to final exam, second subplot
sns.lineplot(x='Student_ID', y='Improvement_Rate', data=df, ax=ax1)
# Plot X=Student_Id and Y=Grades of the practice exam, third subplot
sns.lineplot(x='Student_ID', y='Practice_Exam', data=df, ax=ax2)
plt.show()