/home/pmeng/dl_framework-intel_tensorflow/bazel-bin/tensorflow/python/tools/saved_model_cli convert \
    --dir /tmp/resnet/1538687457 \
    --output_dir /tmp/resnet_trt/1538687457 \
    --tag_set serve \
    tensorrt --precision_mode FP32 --max_batch_size 1 --is_dynamic_op True
    #-dir /tmp/resnet/1538687457 \
    #--output_dir /tmp/resnet_trt/1538687457 \
