FROM python:3.7

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ADD requirements_server.txt .

RUN pip install -r requirements_server.txt

# Change to official seldon-models repo when work https://github.com/SeldonIO/seldon-models/pull/6 merged
ADD https://api.github.com/repos/cliveseldon/seldon-models/git/refs/heads/5_ceserver_update ceserver_version.json
RUN git clone --branch 5_ceserver_update https://github.com/cliveseldon/seldon-models.git && \
    cd seldon-models/servers/cloudevents && \
    pip install -e .

# Fix cloudevents bug: https://github.com/cloudevents/sdk-python/issues/24
RUN git clone --branch 24-extensions https://github.com/ryandawsonuk/sdk-python.git && \
    cd sdk-python && \
    pip install -e .

# Change to official alibi-detect repo when work completed
ADD https://api.github.com/repos/cliveseldon/alibi-detect/git/refs/heads/update_integrations adserver_version.json
RUN git clone --branch update_integrations https://github.com/cliveseldon/alibi-detect.git && \
    cd alibi-detect/integrations/adserver && \
    pip install -e .

ENTRYPOINT ["python", "-m", "adserver"]
