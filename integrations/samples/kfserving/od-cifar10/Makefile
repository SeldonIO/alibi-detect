CIFAR10OD_IMAGE=cifar10od
CIFAR10OD_TAG=0.0.3

vae_outlier_detector:
	gsutil cp -r gs://seldon-models/odcd/vae_outlier_detector .


# REMOVE this when public and update Makefile
tmp/alibi_detect:
	mkdir -p tmp/alibi_detect
	cp ../../../../setup.py tmp/alibi_detect
	cp ../../../../README.md tmp/alibi_detect
	cp -r ../../../../alibi_detect tmp/alibi_detect/

tmp/adserver:
	mkdir -p tmp/
	cp -r ../../../adserver tmp

#
# CIFAR10 Outlier Detector
#

docker-build-cifar10od: clean_tmp tmp/alibi_detect tmp/adserver vae_outlier_detector 
	docker build -f cifar10od.Dockerfile -t seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} .

docker-push-cifar10od:
	docker push seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} 

docker-run-cifar10od:
	docker run --name ${CIFAR10OD_IMAGE} -it --rm -p 8080:8080 seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} --model_name cifar10id --storage_uri ./vae_outlier_detector/


curl-cifar10od-local:
	curl -v localhost:8080/ -d @./input.json 

curl-cifar10od-local-scores:
	curl -v localhost:8080/ -d @./input.json -H "alibi-detect-return-feature-score: true" -H "alibi-detect-return-instance-score: true"

#
# Misc
#

.PHONY: clean_tmp
clean_tmp:
	rm -rf tmp

.PHONY: clean
clean: clean_tmp
	rm -rf vae_outlier_detector
