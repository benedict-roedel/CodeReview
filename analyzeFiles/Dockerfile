FROM  nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -yqq && \
    apt-get dist-upgrade -yqq && \
      apt-get install -yqq \
      git \
      python3 \
      python3-pip \
      python-is-python3 && \
     rm -rf /var/lib/apt/lists/* \

COPY req.txt /tmp/requirements.txt

COPY conda.yaml /tmp/conda.yaml

RUN conda env create -n ppgwas -f /tmp/conda.yml

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install git+https://github.com/PrivateAIM/flame-patterns.git@0.3.1