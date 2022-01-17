FROM ubuntu:20.04
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y python3-pip
EXPOSE 8000
CMD python app.py
