# Description: Tensorflow Serving Caffe servable.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "caffe_session_bundle_config_proto",
    srcs = ["caffe_session_bundle_config.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//google/protobuf:cc_wkt_protos",
        "@tf//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "caffe_session_bundle",
    srcs = ["caffe_session_bundle.cc"],
    hdrs = ["caffe_session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//external:caffe",
        "@tf//google/protobuf:cc_wkt_protos",
        "@tf//google/protobuf:protobuf_lite",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        ":caffe_serving_session",
    ],
)

cc_library(
    name = "caffe_serving_session",
    srcs = ["caffe_serving_session.cc"],
    hdrs = ["caffe_serving_session.h"],
    deps = [
        "//external:caffe",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "caffe_session_bundle_factory",
    srcs = ["caffe_session_bundle_factory.cc"],
    hdrs = ["caffe_session_bundle_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//google/protobuf:cc_wkt_protos",
        "@tf//tensorflow/core:lib",
        ":caffe_session_bundle_config_proto",
        ":caffe_session_bundle",
        ":caffe_serving_session",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_proto",
    ],
)

cc_test(
    name = "caffe_session_bundle_factory_test",
    size = "small",
    srcs = ["caffe_session_bundle_factory_test.cc"],
    data = ["//tensorflow_serving/servables/caffe/example:mnist_pretrained_caffe"],
    deps = [
        "//external:caffe",
        "//tensorflow_serving/servables/caffe/example:mnist_sample",
        "@tf//google/protobuf:cc_wkt_protos",
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        "@tf//tensorflow/core:test",
        "@tf//tensorflow/core:testlib",
        ":caffe_session_bundle_config_proto",
        ":caffe_session_bundle_factory",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "caffe_source_adapter",
    srcs = ["caffe_source_adapter.cc"],
    hdrs = ["caffe_source_adapter.h"],
    visibility = [
        "//visibility:public"
    ],
    deps = [
        "//external:caffe",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":caffe_source_adapter_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
    ],
)

cc_test(
    name = "caffe_source_adapter_test",
    size = "small",
    srcs = ["caffe_source_adapter_test.cc"],
    deps = [
        "//external:caffe",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":caffe_source_adapter",
        ":caffe_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:source_adapter_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:any_ptr",
    ]
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "caffe_source_adapter_proto",
    srcs = ["caffe_source_adapter.proto"],
    cc_api_version = 2,
)
