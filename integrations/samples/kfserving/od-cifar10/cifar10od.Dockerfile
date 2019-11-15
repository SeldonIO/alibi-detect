FROM python:3.7-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN chmod -R a+w /workspace

RUN pip install --upgrade pip

RUN git clone https://github.com/kubeflow/kfserving.git && \
    cd kfserving/python && \
    pip install -e ./kfserving

#RUN git clone https://github.com/seldonio/alibi-detect.git && \
#    cd alibi-detect && \
#    pip install -e .

RUN git clone https://github.com/seldonio/seldon-models.git && \
    cd seldon-models/servers/cloudevents && \
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
