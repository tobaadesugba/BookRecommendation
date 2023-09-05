from surprise import SVD, Dataset, Reader
import pandas as pd
import joblib

data_dir = "C:/Users/TeeJay/PycharmProjects/BookRecommendation/data/processed/ratings_cleaned.csv"  # use cleaned data with no standardization
ratings = pd.read_csv(data_dir, encoding="latin-1")

# Define a Reader object
reader = Reader(rating_scale=(0, 10))

# Create the dataset to be used for building the filter
data = Dataset.load_from_df(ratings[['userID', 'ISBN', 'bookRating']], reader)
print("Dataset created")

# Split the dataset into a training set and a test set
train_set = data.build_full_trainset()

# Define the SVD algorithm object
svd = SVD()

# Train the algorithm on the training set
print("Training begins")
svd.fit(train_set)
print("Training ends")

joblib.dump(svd, 'saved_svd_model.pkl')
print("Model saved")
