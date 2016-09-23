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
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
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
    ],
    deps = [
        ":mnist_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

#
# object detector examples
#

serving_proto_library(
    name = "obj_detector_proto",
    srcs = ["obj_detector.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

py_binary(
    name = "obj_detector_client",
    srcs = [
        "obj_detector_client.py",
        "obj_detector_pb2.py",
        "client_util.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

#
# detector services with Faster R-CNN or SSD backend.
#

config_setting(
    name = "obj_detector_rcnn",
    values = { "define": "detector=rcnn" },
)
config_setting(
    name = "obj_detector_ssd",
    values = { "define": "detector=ssd" },
)

py_binary(
    name = "obj_detector_fetch",
    srcs = ["obj_detector_fetch.py"],
)

cc_binary(
    name = "obj_detector",
    srcs = [
        "obj_detector.cc",
        "obj_detector_utils.cc",
        "obj_detector_utils.h",
        "rpc_utils.h"
    ],
    linkopts = ["-lm"] + select({
        ":obj_detector_ssd": ["-lboost_regex"],
        ":obj_detector_rcnn": []
    }),
    defines = select({
        ":obj_detector_rcnn": ["WITH_RCNN"],
        ":obj_detector_ssd": ["WITH_SSD"]
    }),
    deps = [
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
        ":obj_detector_proto",
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
