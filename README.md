# thisdatedoesnotexist-profile-suggester
 A Flask-based web application that calculates and returns a list of similar user profiles based on their interests, political views, and religion. The server utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity to determine profile similarities.

### Setup

To set up the dating app backend, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python script_name.py
   ```
   The server will start at http://localhost:5002.

### Description

#### 1. Vectorization and Similarity Calculation

- The script uses the `TfidfVectorizer` from scikit-learn to convert profile descriptions into numerical vectors.
- Similarity between profiles is calculated using the cosine similarity metric.

#### 2. User Profile Processing

- The `preprocess_description` function processes user and profile descriptions, considering hobbies, political views, religion, and relationship goals.
- User profiles are enhanced with weights based on political views, favoring similar political perspectives.

#### 3. Finding Similar Profiles

- The `find_similar_profiles` function takes user data and a list of profiles, calculates similarity scores, and returns suggested profiles.
- Profiles are sorted based on similarity scores, considering political views as a factor.

#### 4. API Endpoint

- The `/find-similar-profiles` endpoint accepts POST requests with JSON data containing user information and a list of profiles.
- It returns a JSON response with suggested profiles sorted by similarity.

### API Usage

- **Endpoint:** `/find-similar-profiles`
- **Method:** POST
- **Request Payload:**
  ```json
  {
    "user": {
      "hobbies": [{"name": "hobby1"}, {"name": "hobby2"}],
      "political_view": "center",
      "religion": "agnostic",
      "relationship_goal": {"name": "casual"}
    },
    "profiles": [
      {"id": 1, "hobbies": [...], "political_view": "left", "religion": "atheist", "relationship_goal": {"name": "serious"}},
      ...
    ]
  }
  ```
- **Response Payload:**
  ```json
  {
    "suggested_profiles": [
      {"id": 2, "profile": {...}, "score": 0.85},
      ...
    ]
  }
  ```

### Note

- Ensure that the script is executed in a secure environment, especially in production.
- This script is a basic recommendation system and may require further enhancements for a production-ready application.
- Customize the weights, metrics, and additional features as per the application's requirements.
- Handle exceptions and errors gracefully in a production environment.

### Dependencies

- Flask
- scikit-learn
- waitress (for production-ready server)