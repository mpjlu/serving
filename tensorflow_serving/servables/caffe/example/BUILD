# Tensorflow Serving Caffe session_bundle example.

package(
    default_visibility = ["//tensorflow_serving:internal"],
)

py_binary(
    name = "mnist_caffe_fetch",
    srcs = ["mnist_caffe_fetch.py"],
)

genrule(
    name = "mnist_pretrained_caffe",
    cmd = '''
        rm -rf $(@D)/mnist_pretrained_caffe/00000023;
        $(locations :mnist_caffe_fetch) --version=23 $(@D)/mnist_pretrained_caffe;
        ''', 
    outs = [
        "mnist_pretrained_caffe/00000023/deploy.prototxt",
        "mnist_pretrained_caffe/00000023/weights.caffemodel",
        "mnist_pretrained_caffe/00000023/classlabels.txt",
    ],
    tools = [
        ":mnist_caffe_fetch",
    ],
)

cc_library(
    name = "mnist_sample",
    hdrs = ["mnist_sample.h"]
)
