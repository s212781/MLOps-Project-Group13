# Base image
FROM python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \
    python-pip

WORKDIR /code

COPY requirements.txt code/requirements.txt
COPY setup.py code/setup.py
COPY src/ code/src/
COPY data.dvc src/data.dvc
COPY .dvc/ code/.dvc/
COPY main.py code/main.py


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

RUN pip install -r code/requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "code/main.py"]
