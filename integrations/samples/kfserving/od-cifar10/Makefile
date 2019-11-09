CIFAR10OD_IMAGE=cifar10od
CIFAR10OD_TAG=0.0.1

vae_outlier_detector:
	gsutil cp -r gs://seldon-models/odcd/vae_outlier_detector .


# REMOVE this when public and update Makefile
tmp/odcd:
	mkdir -p tmp/odcd
	cp ../../../../setup.py tmp/odcd
	cp ../../../../README.md tmp/odcd
	cp -r ../../../../odcd tmp/odcd/

tmp/odcdserver:
	mkdir -p tmp/
	cp -r ../../../odcdserver tmp

#
# CIFAR10 Outlier Detector
#

docker-build-cifar10od: clean_tmp tmp/odcd tmp/odcdserver vae_outlier_detector 
	docker build -f cifar10od.Dockerfile -t seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} .

docker-push-cifar10od:
	docker push seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} 

docker-run-cifar10od:
	docker run --name ${CIFAR10OD_IMAGE} -it --rm -p 8080:8080 seldonio/${CIFAR10OD_IMAGE}:${CIFAR10OD_TAG} --model_name cifar10id --storage_uri ./vae_outlier_detector/


curl-cifar10od-local:
	curl -v localhost:8080/ -d @./input.json 

#
# Misc
#

.PHONY: clean_tmp
clean_tmp:
	rm -rf tmp

.PHONY: clean
clean: clean_tmp
	rm -rf vae_outlier_detector
