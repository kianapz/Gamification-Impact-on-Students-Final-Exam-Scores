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

long_csv = pd.read_csv("gamification-software-testing-engagement-performance-test-effectiveness.csv")
short_csv = pd.read_csv("Students_gamification_grades.csv")

#No_access means how many submissions entered for that quiz
short_csv.info()

print("\nAre there any missing points in the dataset?:", short_csv.isnull().values.any())

numerical_features = ['Practice_Exam', 'Final_Exam']
short_csv[numerical_features].hist(bins=30, figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()

categorical_features = ['User']
for feature in categorical_features:
    plt.figure(figsize=(3, 2))
    short_csv[feature].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

#It's a bit... cluttered, supposed to display 'High correlations indicate potential relationships between variables.'
plt.figure(figsize=(6,5))
sns.heatmap(short_csv.corr(), cmap='Blues', annot=True);

#Categorizing users on whether or not they used gamification
def categorize_user(user):
    if user == 0:
        return 'User'
    else:
        return 'Non-user'
short_csv['user_type'] = short_csv['User'].apply(categorize_user)

#Creating a bunch of charts (Plots)
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))
# Plot age vs. charges on the first subplot
sns.lineplot(x='Student_ID', y='Final_Exam', data=short_csv, ax=ax0)
# Plot charges by BMI categories on the second subplot
sns.barplot(x='User', y='Final_Exam', data=short_csv,ax=ax1)
# Plot charges by the number of children on the third subplot
sns.barplot(x='user_type', y='Practice_Exam', data=short_csv,
            order=['User', 'Non-user'], ax=ax2)
# Display the plots
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='User', y='Final_Exam', data=short_csv, hue='user_type', palette='magma', alpha=0.7)
plt.title('Gamification vs. Final Exam Grade by User Types')
plt.show()
