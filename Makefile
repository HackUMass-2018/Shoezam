.PHONY: training-ssh docker-build docker-run docker-notebook

DOCKER_TAG=hackumass-2018/shoezam:latest

# training-ssh opens an ssh connection to the GCP training instance
training-ssh:
	gcloud compute --project "shoezam-2018" ssh --zone "us-east1-b" "ubuntu-training"

# docker-build builds a docker file to run tensorflow with a GPU
docker-build:
	docker build \
		-t "${DOCKER_TAG}" \
		.

# docker-run runs the docker tensorflow gpu docker image
docker-run:
	docker run \
		-it \
		--rm \
		--runtime nvidia \
		-v "${PWD}:/app" \
		"${DOCKER_TAG}" \
		/bin/bash

# docker-notebook runs jupyter notebook
docker-notebook:
	docker run \
		--net host \
		-it \
		--rm \
		--runtime nvidia \
		-v "${PWD}:/app" \
		"${DOCKER_TAG}" \
		jupyter notebook --allow-root
