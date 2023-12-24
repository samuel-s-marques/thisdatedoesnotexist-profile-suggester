from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

vectorizer = None
cosine_similarities = None

political_views_weights = {
    "far left": 1.2,
    "left": 1.1,
    "center-left": 1,
    "center": 0.9,
    "center-right": 0.8,
    "right": 0.7,
    "far right": 0.6,
}


def preprocess_description(data):
    return (
        " ".join([hobby["name"] for hobby in data["hobbies"]])
        + f" {data['political_view']} {data['religion']} {data['relationship_goal']['name']}"
    )


def initialize_vectorizer(descriptions):
    global vectorizer, cosine_similarities
    vectorizer = TfidfVectorizer()
    profile_matrix = vectorizer.fit_transform(descriptions)
    cosine_similarities = linear_kernel(profile_matrix, profile_matrix)


def calculate_similarity_scores(user_data):
    if cosine_similarities is None:
        return []

    user_similarity_scores = list(enumerate(cosine_similarities[0]))

    political_view = user_data["political_view"].lower()
    weight = political_views_weights.get(political_view, 1)

    for i, score in user_similarity_scores:
        user_similarity_scores[i] = (i, score * weight)

    return sorted(user_similarity_scores, key=lambda x: x[1], reverse=True)


def build_profile_data(profiles, sorted_profiles):
    return [
        {
            "id": profiles[i - 1]["id"],
            "profile": profiles[i - 1],
            "score": score,
        }
        for i, score in sorted_profiles
        if i != 0
    ]


def find_similar_profiles(user_data, profiles):
    user_description = preprocess_description(user_data)
    profile_descriptions = [preprocess_description(profile) for profile in profiles]

    combined_descriptions = [user_description] + profile_descriptions

    initialize_vectorizer(combined_descriptions)
    user_similarity_scores = calculate_similarity_scores(user_data)

    sorted_profiles = sorted(user_similarity_scores, key=lambda x: x[1], reverse=True)

    similar_profiles = build_profile_data(profiles, sorted_profiles)

    return similar_profiles


@app.route("/find-similar-profiles", methods=["POST"])
def find_similar_profiles_route():
    try:
        data = request.get_json()
        user_data = data.get("user", {})
        profiles_data = data.get("profiles", [])

        similar_profiles = find_similar_profiles(user_data, profiles_data)

        return jsonify({"suggested_profiles": similar_profiles})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    from waitress import serve

    print(f"Server started at http://localhost:5002")
    serve(app, host="0.0.0.0", port=5002)
