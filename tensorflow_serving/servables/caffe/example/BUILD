

# Description: Tensorflow Serving session_bundle example.

package(
    default_visibility = ["//tensorflow_serving:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

py_binary(
    name = "mnist_caffe_fetch",
    srcs = ["mnist_caffe_fetch.py"],
)

genrule(
    name = "mnist_pretrained_caffe",
    outs = [
        "00000023/deploy.prototxt",
        "00000023/weights.caffemodel",
    ],
    cmd = '''
        rm -rf /tmp/mnist_caffe_fetch;
        $(locations :mnist_caffe_fetch);
        mkdir -p $(@D)/00000023;
        cp -r /tmp/mnist_caffe_fetch/mnist/* $(@D)/00000023; 
        ''', 
    tools = [
        ":mnist_caffe_fetch",
    ],
)

filegroup(
    name = "mnist_sample",
    srcs = ["mnist_sample.h"]
)
