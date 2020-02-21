# CIFAR10 Outlier Detector for Kfserving

A CIFAR10 VAE Outlier Detector. Run the [notebook demo](cifar10_outlier.ipynb) to test.

To check OD locally, first run the same pip install commands that cifar10od.Dockerfile does. Then:

1) Start docker for elasticsearch following README in seldon-core/seldon-request-logger
2) Do `make run_local_cifar` in seldon-core/seldon-request-logger
3) From the same directory as this README (`alibi-detect/integrations/samples/od-cifar10`) do `make local-run`
4) Do `m̀ake curl-cifar10od-local`
5) The logs for the request logger give the index doc type and id for the inserted doc. You can take this string and run `curl -X GET http://localhost:9200/<string>` to see the indexed doc