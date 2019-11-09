FROM python:3.6-slim

WORKDIR /workspace

RUN chmod -R a+w /workspace

RUN pip install --upgrade pip

RUN git clone https://github.com/kubeflow/kfserving.git && \
    cd kfserving/python && \
    pip install -e ./kfserving

#RUN git clone https://github.com/seldonio/odcd.git && \
#    cd odcd && \
#    pip install -e .

RUN git clone https://github.com/seldonio/seldon-models.git && \
    cd seldon-models/servers/cloudevents && \
    pip install -e .


COPY tmp tmp

RUN cd tmp/odcd && \
    pip install -e .

RUN cd tmp/odcdserver && \
    pip install -e .

COPY ad_vae_mnist ad_vae_mnist

ENTRYPOINT ["python", "-m", "odcdserver"]

