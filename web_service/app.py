from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model at the start
model_dir = "C:/Users/TeeJay/PycharmProjects/BookRecommendation/src/models/saved_svd_model.pkl"
model = joblib.load(model_dir)


@app.route('/')
def index():
    return "Book Recommendation System"


@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.json
    user_id = data['user_id']
    n_recommendations = data.get('n_recommendations', 5)

    # Assuming you have a function to get recommendations using the SVD model
    recommended_books = get_recommendations_for_user(user_id, n_recommendations)

    return jsonify(recommended_books)


def get_recommendations_for_user(user_id, n_recommendations):
    # Predict ratings for all books for the given user
    predictions = [model.predict(user_id, book_id) for book_id in all_book_ids]

    # Sort predictions in descending order of rating
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Extract the top N book recommendations
    recommended_books = [{"book_id": pred.item, "predicted_rating": pred.est} for pred in sorted_predictions[:n_recommendations]]

    return recommended_books


if __name__ == "__main__":
    app.run(port=5000, debug=True)  # 'debug=True' is for development only!
