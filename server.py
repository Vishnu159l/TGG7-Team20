from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

@app.route('/get_result')
def get_result():
    try:
        with open('C:\\Users\\Akshay Prakash\\Documents\\Semester 4\\Hackathon\\tmp\\result.txt', 'r') as f:
            result = f.read().strip()
        return jsonify({"result": result})
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000)