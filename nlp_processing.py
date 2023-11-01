
import os
import pandas as pd
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("Tweets.csv")

# Drop unnecessary columns
data.dropna(inplace = True)

# Function to clean text
def clean(text):
    if isinstance (text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data["text"] = data["text"].apply(clean)

# Remove rows with missing values
data.dropna(inplace=True)

# Initialize the stemmer
stemmer = SnowballStemmer("english")

# Apply stemming to the text
data["new_text"] = data["text"].apply(lambda x: [stemmer.stem(i) for i in x])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data["text"], data["sentiment"], test_size=0.2, random_state=42)

# Define the path for the output data as the same folder
notebook_dir = os.path.abspath('')
PROCESSED_DATA_DIR = notebook_dir

# Save the dataframes as CSV files
x_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "x_train.csv"))
x_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "x_test.csv"))
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"))
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"))

