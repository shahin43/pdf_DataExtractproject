FROM ubuntu:18.04

RUN mkdir /app_data_extract
WORKDIR /app_data_extract

ADD . /app_data_extract

RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    python3 \
    #python-setuptools \
    python3-pip \
    poppler-utils \
    awscli \
    && apt-get clean \
    && apt-get autoremove

RUN python3 -m pip install --no-cache-dir --upgrade \
        setuptools \
        wheel

RUN python3 -m pip install virtualenv

RUN virtualenv ../ocrevn
RUN ../ocrevn/bin/pip install -r ./requirements.txt

CMD tail -f /dev/null
