SIGNS_IMAGE=signs
SIGNS_TAG=0.0.1
SIGNSAD_IMAGE=signsad
SIGNSAD_TAG=0.0.1

model_signs:
	gsutil cp -r gs://seldon-models/odcd/model_signs .

ad_vae_signs:
	gsutil cp -r gs://seldon-models/odcd/ad_vae_signs .

gtsrb-german-traffic-sign.zip:
	gsutil cp gs://seldon-datasets/traffic/gtsrb-german-traffic-sign.zip .

traffic_data: gtsrb-german-traffic-sign.zip
	mkdir -p traffic_data
	cd traffic_data && unzip ../gtsrb-german-traffic-sign.zip

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
# SIGNS Model
#

docker-build-signs: tmp/odcd model_signs
	docker build -f signs.Dockerfile -t seldonio/${SIGNS_IMAGE}:${SIGNS_TAG} .

docker-run-signs:
	docker run --name ${SIGNS_IMAGE} -it --rm -p 8080:8080 seldonio/${SIGNS_IMAGE}:${SIGNS_TAG} --model_name signs --model_dir ./model_signs/

docker-push-signs:
	docker push seldonio/${SIGNS_IMAGE}:${SIGNS_TAG}

curl-signs-local:
	curl -v http://0.0.0.0:8080/v1/models/signs:predict -d @./signs.json

#
# SIGNS Adversarial Detector
#

docker-build-signsad: tmp/odcd tmp/odcdserver ad_vae_signs
	docker build -f signsad.Dockerfile -t seldonio/${SIGNSAD_IMAGE}:${SIGNSAD_TAG} .

docker-run-signsad:
	docker run --name ${SIGNSAD_IMAGE} --rm -p 8080:8080 seldonio/${SIGNSAD_IMAGE}:${SIGNSAD_TAG} --model_name signsad --storage_uri ./ad_vae_signs/

docker-push-signsad:
	docker push seldonio/${SIGNSAD_IMAGE}:${SIGNSAD_TAG}

curl-signsad-local:
	curl -v http://0.0.0.0:8080/ -d @./signs.json

#
# Misc
#

.PHONY: clean_tmp
clean_tmp:
	rm -rf tmp

.PHONY: clean
clean: clean_tmp
	rm -rf ad_vae_signs
	rm -rf model_signs
	rm -rf traffic_data
	rm gtsrb-german-traffic-sign.zip


