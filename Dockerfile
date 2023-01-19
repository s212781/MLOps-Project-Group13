# # Base image
FROM python:3.8-slim

# install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-setuptools \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN pip3 install --upgrade pip
# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git

#RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /root

# Make sure gsutil will use the default service account
# RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY requirements.txt requirements.txt
COPY src/ src/
COPY .dvc/config .dvc/config
COPY data_lite.dvc data_lite.dvc
COPY .git .git
COPY entrypoint.sh entrypoint.sh
COPY models/ models/
COPY main.py main.py
RUN pip3 install multidict
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install wandb
RUN pip3 install dvc
RUN pip3 install dvc[gs]
RUN ls

ENTRYPOINT ["sh", "entrypoint.sh"]

# # # Base image
# FROM ubuntu:18.04

# # install python
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.8 \
#     python3-pip \
#     python3-setuptools \
#     && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /

# COPY requirements.txt requirements.txt
# COPY src/ src/
# COPY data.dvc data.dvc
# COPY .dvc/ .dvc/
# COPY main.py main.py
# COPY model_v1_0.pth.dvc model_v1_0.pth.dvc
# COPY data_lite data_lite
# RUN python3 -m pip install -U pip
# RUN pip3 install -r requirements.txt --no-cache-dir

# ENTRYPOINT ["python3", "-u", "main.py"]

