from flask import Flask, request, jsonify
import tensorflow as tf
from model import model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class DummyArgs(object):
    pass

@app.route('/api/summary', methods=['POST'])
def summary():
    if request.method == 'POST':
        data = request.json
        result = summary(data)
        return jsonify(result)

def summary(data):
    summary, scores = sever_model.input_prediction(data['article'], data['summary'])
    return {'summary': summary, 'scores': scores}

if __name__ == '__main__':

    sess = tf.Session()
    args = DummyArgs()
    args.action = 'test'
    args.load = ''
    args.datatype = ''
    sever_model = model(sess, args)
    sever_model.server_init()

    app.debug = False
    app.run()