FROM python:3.7-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext-dev && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

