config_setting(
    name = "caffe_python_layer",
    values = {
        "define": "caffe_python_layer=ON"
    },
    visibility = ["//visibility:public"],
)

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

cc_library(
    name = "openblas_prelude",
    hdrs = ["openblas_prelude.h"],
)

cc_library(
    name = "python_prelude",
    hdrs = ["python_prelude.h"],
)

filegroup(
    name = "pycaffe_overrides",
    srcs = ["__init__.py"]
)