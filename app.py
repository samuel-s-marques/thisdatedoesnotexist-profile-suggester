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


def find_similar_profiles(user_data, profiles):
    similar_profiles = []

    user_description = (
        " ".join([hobby["name"] for hobby in user_data["hobbies"]])
        + f" {user_data['political_view']} {user_data['religion']} {user_data['relationship_goal']['name']}"
    )
    profile_descriptions = [
        " ".join([hobby["name"] for hobby in profile["hobbies"]])
        + f" {profile['political_view']} {profile['religion']} {profile['relationship_goal']['name']}"
        for profile in profiles
    ]

    combined_descriptions = [user_description] + profile_descriptions

    global vectorizer, cosine_similarities
    vectorizer = TfidfVectorizer()
    profile_matrix = vectorizer.fit_transform(combined_descriptions)
    cosine_similarities = linear_kernel(profile_matrix, profile_matrix)

    user_similarity_scores = list(enumerate(cosine_similarities[0]))

    for i, score in user_similarity_scores:
        political_view = user_data["political_view"].lower()
        weight = political_views_weights.get(political_view, 1)
        user_similarity_scores[i] = (i, score * weight)

    sorted_profiles = sorted(user_similarity_scores, key=lambda x: x[1], reverse=True)

    similar_profiles = [
        {
            "id": profiles[i - 1]["id"],
            "profile": profiles[i - 1],
            "score": score,
        }
        for i, score in sorted_profiles
        if i != 0
    ]

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
    app.run(debug=True)
