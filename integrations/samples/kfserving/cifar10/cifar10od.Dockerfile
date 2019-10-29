FROM python:3.7-slim

COPY . .
RUN pip install --upgrade pip
RUN cd tmp/odcd && pip install -e .
RUN cd tmp/cloudevents && pip install -e .
RUN cd tmp/odcdserver && pip install -e .
RUN cd cifar10od && pip install -e .
ENTRYPOINT ["python", "-m", "cifar10od"]
