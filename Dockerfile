FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install flask

WORKDIR /app
#ENTRYPOINT /bin/sh
