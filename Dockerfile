# # Base image
FROM ubuntu:18.04

# install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# WORKDIR /code

# COPY requirements.txt code/requirements.txt
# COPY setup.py code/setup.py
# COPY src/ code/src/
# COPY data.dvc src/data.dvc
# COPY .dvc/ code/.dvc/
# COPY main.py code/main.py
# COPY init.sh /code/init.sh

# RUN apt-get -y update; apt-get -y install curl

# # Downloading gcloud package
# RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# # Installing the package
# RUN mkdir -p /usr/local/gcloud \
#   && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
#   && /usr/local/gcloud/google-cloud-sdk/install.sh 

# #Adding the package path to local
# ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# RUN gcloud auth configure-docker

# RUN pip3 install -r code/requirements.txt --no-cache-dir

# ENTRYPOINT ["python", "-u", "code/inti.sh"]

# FROM python:3.8-slim

# # install python 
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY main.py main.py
# COPY src/ src/
# COPY data.dvc data.dvc
# COPY model_v1_0.pth.dvc model_v1_0.pth.dvc
# COPY .dvc .dvc

# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir

# ENTRYPOINT ["python", "-u", "main.py"]

