from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from analysis import exploratory, kmeans
from pymongo import MongoClient
from flask_cors import CORS

# Configs
app = Flask(__name__)
CORS(app)
PORT = 4000
HOST = "0.0.0.0"

# Connect to mongodb
MONGO_URI = "mongodb://mongo:27017/"  # Port on docker
client = MongoClient(MONGO_URI)

# Store of my db
db = client["database"]

# Store collection in the db
collection = db["clients"]

# Use the functions
exploratory_analysis = exploratory.analysis
kmeans_algorithm = kmeans.kmeans_algorithm


@app.route('/')
def main():

    return "<h1> Hola </h1>"


@app.route('/kmeans/<int:client_id>', methods=['GET'])
def kmeans(client_id):
    if request.method == 'GET':
        clients = collection.find()
        return kmeans_algorithm(client_id, clients)


@app.route('/exploratory-analysis', methods=['GET'])
def analysis():
    if request.method == 'GET':
        clients = collection.find()
        return exploratory_analysis(clients)


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)
