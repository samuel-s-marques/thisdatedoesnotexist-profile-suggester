from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/find-similar-profiles', methods=['POST'])
def find_similar_profiles_route():
    try:
        return jsonify({'message': 'Hello World!'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)