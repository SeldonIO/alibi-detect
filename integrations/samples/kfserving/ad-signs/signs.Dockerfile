FROM python:3.6

WORKDIR /workspace

RUN chmod -R a+w /workspace

RUN pip install --upgrade pip

RUN git clone https://github.com/kubeflow/kfserving.git && \
    cd kfserving/python && \
    pip install -e ./kfserving

#RUN git clone https://github.com/seldonio/odcd.git && \
#    cd odcd && \
#    pip install -e .

COPY tmp tmp
RUN cd tmp/odcd && \
    pip install -e .

COPY model_signs model_signs
COPY signs.py signs.py

ENTRYPOINT ["python", "signs.py"]

