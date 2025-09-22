
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dataset (update path if needed)
df = pd.read_csv("Titanic-Dataset.csv")

# 2️⃣ Preview dataset
print("First 5 rows of dataset:")
print(df.head())

# 3️⃣ Shape of dataset
print("\nShape of dataset:", df.shape)

# 4️⃣ Info about dataset
print("\nDataset info:")
print(df.info())

# 5️⃣ Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nPercentage of missing values per column:")
print((df.isnull().sum() / len(df)) * 100)

# 6️⃣ Statistical summary
print("\nStatistical summary:")
print(df.describe())

# 7️⃣ Handle missing values safely
df['Age'] = df['Age'].fillna(df['Age'].median())        # Fill Age with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill Embarked with mode

# 8️⃣ Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# 9️⃣ Encode categorical variables
# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# Label encoding for Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 10️⃣ Scale numerical columns
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 11️⃣ Visualize boxplots to detect outliers
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# 12️⃣ Remove outliers in Fare (keep values below 99th percentile)
upper_limit = df['Fare'].quantile(0.99)
df = df[df['Fare'] <= upper_limit]

# 13️⃣ Final cleaned dataset preview
print("\nCleaned dataset preview (first 5 rows):")
print(df.head())
print("\nShape of cleaned dataset:", df.shape)
