from flask import Flask, send_from_directory, request
import random
import uuid
import tempfile

import predict_v12 as predict

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def sendHomepage():
	if request.method == "GET":
		return app.send_static_file("index.html")
	elif request.method == "POST":
		imagePath = tempfile.NamedTemporaryFile("w").name
		request.files["image"].save(imagePath)

		print(imagePath)
		pred = predict.is_shoe(imagePath)
		if pred:
			return app.send_static_file("shoe.html")
		else:
			return app.send_static_file("notshoe.html")


@app.route("/static/<path:path>")
def sendStaticFiles(path):
	return send_from_directory(path)
