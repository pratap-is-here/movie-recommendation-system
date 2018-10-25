
# coding: utf-8

import pickle

from flask import Flask, jsonify
from flask import request
from kernel import pikloader


app = Flask(__name__)
@app.route('/')
def home():
	return 'hi there'

#defining a /hello route for only post requests
@app.route('/<string:movie_name>', methods=['GET'])
def index(movie_name):

    return jsonify({'movies': pikloader(movie_name)})

if __name__ == '__main__':
    app.run(debug=True)

