Titanic Dataset – Data Cleaning 

This project walks you through cleaning and preparing the Titanic dataset for analysis or machine learning. Using Python and libraries like pandas, seaborn, matplotlib, and scikit-learn, we handle missing data, encode categorical values, scale numerical features, and detect outliers — all in one script.

📁 About the Dataset

File Name: Titanic-Dataset.csv

Source: Typically from Kaggle’s Titanic competition
 or similar.

What’s Inside: Passenger details like name, age, gender, ticket fare, port of embarkation, and whether they survived.

🔧 What This Script Does
✅ Loads and Explores the Data

Reads the dataset using pandas.

Displays the first few rows to get a feel for the data.

Shows the shape, column info, missing values, and basic statistics.

🧹 Cleans Missing Data

Fills missing Age values using the median.

Fills missing Embarked values using the most common port (mode).

Drops the Cabin column since it’s mostly empty.

🧬 Encodes Categorical Columns

Converts Sex into numeric values: male → 0, female → 1.

Uses one-hot encoding for the Embarked column (excluding the first to avoid redundancy).

📊 Scales Numerical Features

Standardizes Age and Fare values using StandardScaler to bring them onto the same scale.

📉 Detects and Removes Outliers

Uses boxplots to visually spot outliers in Age and Fare.

Removes extreme outliers in Fare by trimming values above the 99th percentile.

🖥️ Requirements

Make sure you have these Python packages installed:

pip install pandas matplotlib seaborn scikit-learn

🧪 How to Run It

Download or place the Titanic-Dataset.csv in the same folder as your script.

Run the script using:

python titanic_cleaning.py

📈 What You'll See

Initial and cleaned versions of the dataset printed in the terminal.

Boxplots showing Age and Fare before outlier removal.

The shape of the dataset before and after cleaning.


📤 Final Output

By the end of this script, you’ll have:

A cleaned dataset with no missing values in critical columns.

Categorical variables encoded.

Scaled numerical values.

Outliers in Fare removed for better model performance.
