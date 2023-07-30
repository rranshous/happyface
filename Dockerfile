FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install pip python3 -y
RUN pip install opencv-python-headless

WORKDIR /app
COPY ./ /app

VOLUME /data
