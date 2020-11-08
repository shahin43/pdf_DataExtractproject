FROM clearlinux/tesseract-ocr:latest

RUN mkdir /app_data_extract
WORKDIR /app_data_extract

ADD . /app_data_extract

ARG swupd_args
# Move to latest Clear Linux release to ensure
# that the swupd command line arguments are
# correct
RUN swupd update --no-boot-update $swupd_args

RUN swupd bundle-add python3-tcl
RUN swupd bundle-add cloud-control
RUN swupd bundle-add poppler
    # rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    # rm -rf /tmp/* /var/tmp/*

RUN python3 -m pip install --no-cache-dir --upgrade \
        setuptools \
        wheel

RUN python3 -m pip install virtualenv

## this step is required to move trained languages to tessdata directory
COPY eng.traineddata /usr/share/tessdata/
COPY ara.traineddata /usr/share/tessdata/
COPY osd.traineddata /usr/share/tessdata/

RUN virtualenv ../ocrevn
RUN ../ocrevn/bin/pip install -r ./requirements.txt

CMD tail -f /dev/null
