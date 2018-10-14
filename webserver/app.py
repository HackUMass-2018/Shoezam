from flask import Flask, send_from_directory, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def sendHomepage():
	if request.method == "GET":
		return app.send_static_file("index.html")
	elif request.method == "POST":
		image = request.files["image"]
		# TODO do something with the image
		return "Maybe shoe"


@app.route("/static/<path:path>")
def sendStaticFiles(path):
	return send_from_directory(path)
