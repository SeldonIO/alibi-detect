CIFAR10OD_IMAGE=cifar10od
CIFAR10OD_TAG=0.0.5
LOCAL_DIR:=$(shell pwd)

cifar10:
	gsutil cp -r gs://seldon-models/alibi-detect/od/OutlierVAE/cifar10 .

.PHONY: run-od
run-od: cifar10
	python -m adserver --model_name cifar10od --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.outlier --storage_uri file://$(LOCAL_DIR)/cifar10/ --event_source http://localhost:8080 OutlierDetector
#--reply_url http://localhost:2222

.PHONY: run-od-remote-model
run-od-remote-model:
	python -m adserver --model_name cifar10od --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.outlier --storage_uri gs://seldon-models/alibi-detect/od/OutlierVAE/cifar10 --event_source http://localhost:8080 OutlierDetector

curl-od:
	curl -v localhost:8080/ -d @./input.json -H "ce-namespace: default" -H "ce-modelid: cifar10"

curl-od-scores:
	curl -v localhost:8080/ -d @./input.json -H "Alibi-Detect-Return-Feature-Score: true" -H "Alibi-Detect-Return-Instance-Score: true"


docker-build: 
	docker build -f cifar10od.Dockerfile -t seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} .

docker-push:
	docker push seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} 

docker-run:
	docker run --name ${CIFAR10OD_IMAGE} -it --rm -p 8080:8080 seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} --model_name cifar10od --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.outlier --storage_uri gs://seldon-models/alibi-detect/od/OutlierVAE/cifar10 --event_source http://localhost:8080 OutlierDetector

