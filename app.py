from flask import Flask, send_from_directory, request
import tempfile

import neural_network.libmodel as libmodel
import neural_network.predict as predict

# Load trained model
model_graph = libmodel.create_graph()

with model_graph.as_default():
	model = libmodel.load_trained_model()

# Create flask app
app = Flask(__name__)

# Define routes
@app.route("/", methods=["GET", "POST"])
def route_homepage():
	if request.method == "GET":
		return app.send_static_file("index.html")
	elif request.method == "POST":
		# Save uploaded file
		image_path = tempfile.NamedTemporaryFile("w").name
		request.files["image"].save(image_path)

		# Run prediction
		with model_graph.as_default():
			is_shoe = predict.is_shoe(model, image_path)

		print("Is shoe: {}".format(is_shoe))

		# Return HTML page based on prediction result
		if is_shoe:
			return app.send_static_file("shoe.html")
		else:
			return app.send_static_file("notshoe.html")


@app.route("/static/<path:path>")
def route_static_files(path):
	return send_from_directory(path)
