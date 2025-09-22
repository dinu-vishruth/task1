
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset 
df = pd.read_csv("Titanic-Dataset.csv")

# Preview dataset
print("First 5 rows of dataset:")
print(df.head())

#  Shape of dataset
print("\nShape of dataset:", df.shape)

# Info about dataset
print("\nDataset info:")
print(df.info())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nPercentage of missing values per column:")
print((df.isnull().sum() / len(df)) * 100)

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Handle missing values safely
df['Age'] = df['Age'].fillna(df['Age'].median())        # Fill Age with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill Embarked with mode

#  Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Encode categorical variables
# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# Label encoding for Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

#Scale numerical columns
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Visualize boxplots to detect outliers
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# Remove outliers in Fare (keep values below 99th percentile)
upper_limit = df['Fare'].quantile(0.99)
df = df[df['Fare'] <= upper_limit]

# Final cleaned dataset preview
print("\nCleaned dataset preview (first 5 rows):")
print(df.head())
print("\nShape of cleaned dataset:", df.shape)

