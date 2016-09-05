# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

config_setting(
    name = "use_caffe",
    values = {"define": "runtime=caffe"},
)

serving_proto_library(
    name = "mnist_inference_proto",
    srcs = ["mnist_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
)

py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    deps = [
        ":mnist_input_data",
        "//tensorflow_serving/session_bundle:exporter",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

cc_binary(
    name = "mnist_inference",
    srcs = [
        "mnist_inference.cc",
    ],
    linkopts = ["-lm"],
    defines = select({
        ":use_caffe": ["USE_CAFFE"],
        "//conditions:default": []
    }),
    deps = [
        ":mnist_inference_proto",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
    ] + select({
        ":use_caffe": [
            "//tensorflow_serving/servables/caffe:caffe_session_bundle_config_proto",
            "//tensorflow_serving/servables/caffe:caffe_session_bundle_factory",
            "//tensorflow_serving/servables/caffe:caffe_session_bundle",
            "//tensorflow_serving/servables/caffe:caffe_signature",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:core_cpu",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
            "@org_tensorflow//tensorflow/core:tensorflow",
            "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
            "//tensorflow_serving/servables/tensorflow:session_bundle_factory",        
            "//tensorflow_serving/session_bundle:signature",
        ]
    }),
)

cc_binary(
    name = "mnist_inference_2",
    srcs = [
        "mnist_inference_2.cc",
    ],
    linkopts = ["-lm"],
    defines = select({
        ":use_caffe": ["USE_CAFFE"],
        "//conditions:default": []
    }),
    deps = [
        ":mnist_inference_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/session_bundle:signature",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
    ] + select({
        ":use_caffe": [
            "//tensorflow_serving/servables/caffe:caffe_session_bundle",
            "//tensorflow_serving/servables/caffe:caffe_simple_servers",
            "//tensorflow_serving/servables/caffe:caffe_signature",
        ],
        "//conditions:default": [
            "//tensorflow_serving/session_bundle",
            "//tensorflow_serving/session_bundle:manifest_proto",
            "//tensorflow_serving/servables/tensorflow:simple_servers",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
            "@org_tensorflow//tensorflow/core:tensorflow",

        ]
    }),
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
        "mnist_inference_pb2.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

serving_proto_library(
    name = "inception_inference_proto",
    srcs = ["inception_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "//tensorflow_serving/session_bundle:exporter",
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

cc_binary(
    name = "inception_inference",
    srcs = [
        "inception_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":inception_inference_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
        "inception_inference_pb2.py",
    ],
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)


#
# faster-rcnn example
#

# Downloads a py-faster-rcnn repo from github
py_binary(
    name = "rcnn_fetch",
    srcs = ["rcnn_fetch.py"],
)

py_binary(
    name = "rcnn_setup",
    srcs = ["rcnn_setup.py"],
)

serving_proto_library(
    name = "rcnn_detector_proto",
    srcs = ["rcnn_detector.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

cc_binary(
    name = "rcnn_detector",
    srcs = ["rcnn_detector.cc", "rcnn_utils.cc", "rcnn_utils.h"],
    linkopts = ["-lm"],
    deps = [
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
        ":rcnn_detector_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/session_bundle:signature",
        "//tensorflow_serving/servables/caffe:caffe_session_bundle",
        "//tensorflow_serving/servables/caffe:caffe_simple_servers",
        "//tensorflow_serving/servables/caffe:caffe_signature",
    ]
)

py_binary(
    name = "rcnn_client",
    srcs = [
        "rcnn_client.py",
        "rcnn_detector_pb2.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)