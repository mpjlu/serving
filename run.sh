bazel build -c opt --config=cuda  --define=detector=ssd --define=caffe_flavour=ssd --verbose_failures    //tensorflow_serving/example:obj_detector 2>&1 | tee log.txt
