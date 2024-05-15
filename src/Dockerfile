FROM ubuntu:24.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /ner

COPY requirements.txt .

# Install required packages
RUN pip install -r requirements.txt --break-system-packages

COPY . .