FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install flask
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /app
#ENTRYPOINT /bin/sh
