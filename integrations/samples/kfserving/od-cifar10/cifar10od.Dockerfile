FROM python:3.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN chmod -R a+w /workspace

RUN pip install --upgrade pip

ADD https://api.github.com/repos/kubeflow/kfserving/git/refs/heads/master version.json
RUN git clone https://github.com/kubeflow/kfserving.git && \
    cd kfserving/python && \
    pip install -e ./kfserving

#RUN git clone https://github.com/seldonio/alibi-detect.git && \
#    cd alibi-detect && \
#    pip install -e .

ADD https://api.github.com/repos/ryandawsonuk/seldon-models/git/refs/heads/1393-od-reqlogging version.json
RUN git clone --branch 1393-od-reqlogging https://github.com/ryandawsonuk/seldon-models.git && \
    cd seldon-models/servers/cloudevents && \
    pip install -e .

RUN git clone --branch 24-extensions https://github.com/ryandawsonuk/sdk-python.git && \
    cd sdk-python && \
    pip install -e .

COPY tmp tmp

RUN cd tmp/alibi_detect && \
    pip install -e .

RUN cd tmp/adserver && \
    pip install -e .

COPY cifar10od cifar10od

RUN cd cifar10od && pip install -e .

COPY vae_outlier_detector vae_outlier_detector

ENTRYPOINT ["python", "-m", "cifar10od"]
