#script hosted on https://rdhirschel.pythonanywhere.com/
from flask import Flask, jsonify, request, make_response, after_this_request
from flask_cors import CORS, cross_origin
from DL3 import *
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/classify', methods=['POST', 'OPTIONS'])
@cross_origin(origin='*', headers=['Content-Type','Authorization'])
def classify_image():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    elif request.method == 'POST':
        # Load the model
        model = DLModel()
        model.load_weights("/home/Rdhirschel/mysite/saved_weights 80.71%", ["relu", "trim_sigmoid", "trim_tanh", "trim_softmax"], "categorical_cross_entropy")

        # Get the image data from the request
        data = request.get_json()
        image_array = np.array(data['data'])
        image_array = image_array.reshape(32*32*3, 1) / 255.0 - 0.5
        prediction = model.predict(image_array)
        classNames = ["dandelion", "iris", "rose", "sunflower", "tulip"]
        label = classNames[np.argmax(prediction)]

        response = make_response(jsonify({'label': label}))
        return _corsify_actual_response(response)

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response