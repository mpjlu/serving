cc_library(
    name = "caffe",
    srcs = ["lib/libcaffe.so.1", "lib/libproto.a"],
    hdrs = glob(["include/**"]),
    includes = ["include/"],
    defines = ["CPU_ONLY"],
    linkopts = [],
    visibility = ["//visibility:public"]
)