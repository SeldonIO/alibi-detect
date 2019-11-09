

docker-build-cifar10od: clean_tmp
	mkdir -p tmp/odcd
	cd tmp && git clone https://github.com/SeldonIO/utils.git
	cp ../../../../setup.py tmp/odcd
	cp -r ../../../../odcd tmp/odcd/
	cp -r ../../../odcdserver tmp
	mkdir examples
	cp -r ../../../../examples/vae_outlier_detector examples
	docker build -f cifar10od.Dockerfile -t seldonio/cifar10od:0.0.1 .

clean_tmp:
	rm -rf tmp
	rm -rf examples

docker-push-cifar10od:
	docker push seldonio/cifar10od:0.0.1

