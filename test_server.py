from flask import Flask
import time

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello"

if __name__ == "__main__":
    print("Starting test server...")
    app.run(host="127.0.0.1", port=5002)
