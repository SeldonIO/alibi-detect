LOCAL_DIR:=$(shell pwd)

cifar10:
	gsutil cp -r gs://seldon-models/alibi-detect/cd/ks/cifar10 .

.PHONY: run-cd
run-cd: cifar10
	python -m adserver --model_name cifar10cd --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.drift --storage_uri file://$(LOCAL_DIR)/cifar10/ --event_source http://localhost:8080 DriftDetector --drift_batch_size=2

.PHONY: run-cd-remote-model
run-cd-remote-model:
	python -m adserver --model_name cifar10cd --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.drift --storage_uri gs://seldon-models/alibi-detect/cd/ks/cifar10 --event_source http://localhost:8080 DriftDetector --drift_batch_size=2

curl-cd:
	curl -v localhost:8080/ -d @./input.json -H "ce-namespace: default" -H "ce-modelid: cifar10"

docker-run:
	docker run --name cifar10cd  -it --rm -p 8080:8080 seldonio/alibi-detect-server:0.0.1 --model_name cifar10cd --http_port 8080 --protocol tensorflow.http --event_type org.kubeflow.serving.inference.drift --storage_uri gs://seldon-models/alibi-detect/cd/ks/cifar10 --event_source http://localhost:8080 DriftDetector --drift_batch_size=2

