from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

@app.route('/hello',methods=['POST','GET'])
def hello():
    message = request.get_json(force = True)
    name = message['name']
    response = {
        'greeting': 'Sup,' + name + '?'
    }
    return jsonify(response)
