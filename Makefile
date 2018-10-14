.PHONY: training-ssh docker-build docker-run docker-shell docker-notebook docker-web docker-train docker-predict

MAKE=make
DOCKER_TAG=hackumass-2018/shoezam:latest

# training-ssh opens an ssh connection to the GCP training instance
training-ssh:
	gcloud compute --project "shoezam-2018" ssh --zone "us-east1-b" "ubuntu-training"

# docker-build builds a docker file to run tensorflow with a GPU
docker-build:
	docker build \
		-t "${DOCKER_TAG}" \
		.

# docker-run runs a command in the docker tensorflow gpu docker container
# args:
#	- EXEC: command to execute
docker-run:
	if [ -z "${EXEC}" ]; then echo "EXEC argument must be provided"; exit 1; fi
	docker run \
		--net host \
		-it \
		--rm \
		--runtime nvidia \
		-v "${PWD}:/app" \
		"${DOCKER_TAG}" \
		${EXEC}

# docker-shell starts a bash terminal in the docker container
docker-shell:
	${MAKE} docker-run EXEC="/bin/bash"

# docker-notebook runs jupyter notebook in the docker container
docker-notebook:
	${MAKE} docker-run EXEC="jupyter notebook"

# docker-web starts the flask server in the docker container
docker-web:
	${MAKE} docker-run EXEC="\"/usr/local/bin/flask\" run -h 0.0.0.0"

# docker-train trains the model in the docker container
docker-train:
	${MAKE} docker-run EXEC="./neural_network/train.py"

# docker-predict runs the model prediction tests in the docker container
docker-predict:
	${MAKE} docker-run EXEC="./neural_network/predict.py"
