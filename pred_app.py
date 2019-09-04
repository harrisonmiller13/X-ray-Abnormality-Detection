from flask import Flask
app = Flask(__name__)

@app.route('/isthisweird')
def running():
    return 'Flask is running'