MNIST_IMAGE=mnist
MNIST_TAG=0.0.3
MNISTAD_IMAGE=mnistad
MNISTAD_TAG=0.0.3

model_mnist:
	gsutil cp -r gs://seldon-models/odcd/model_mnist .

ad_vae_mnist:
	gsutil cp -r gs://seldon-models/odcd/ad_vae_mnist .

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
# MNIST Model
#

docker-build-mnist: tmp/alibi_detect model_mnist
	docker build -f mnist.Dockerfile -t seldonio/${MNIST_IMAGE}:${MNIST_TAG} .

docker-run-mnist:
	docker run --name ${MNIST_IMAGE} -it --rm -p 8080:8080 seldonio/${MNIST_IMAGE}:${MNIST_TAG} --model_name mnist --model_dir ./model_mnist/

docker-push-mnist:
	docker push seldonio/${MNIST_IMAGE}:${MNIST_TAG}

curl-mnist-local:
	curl -v http://0.0.0.0:8080/v1/models/mnist:predict -d @./mnist.json

#
# MNIST Adversarial Detector
#

docker-build-mnistad: tmp/alibi_detect tmp/adserver ad_vae_mnist
	docker build -f mnistad.Dockerfile -t seldonio/${MNISTAD_IMAGE}:${MNISTAD_TAG} .

docker-run-mnistad:
	docker run --name ${MNISTAD_IMAGE} --rm -p 8080:8080 seldonio/${MNISTAD_IMAGE}:${MNISTAD_TAG} --model_name mnistad --storage_uri ./ad_vae_mnist/

docker-push-mnistad:
	docker push seldonio/${MNISTAD_IMAGE}:${MNISTAD_TAG}

curl-mnistad-local:
	curl -v http://0.0.0.0:8080/ -d @./mnist.json

#
# Misc
#

.PHONY: clean_tmp
clean_tmp:
	rm -rf tmp

.PHONY: clean
clean: clean_tmp
	rm -rf ad_vae_mnist
	rm -rf model_mnist


