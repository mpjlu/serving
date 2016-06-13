load("@//third_party:caffe.bzl", "if_cuda")
load("@org_tensorflow//third_party/gpus/cuda:platform.bzl",
     "cuda_sdk_version",
     "cudnn_library_path",
    )

package(default_visibility = ["//visibility:public"])

CAFFE_WELL_KNOWN_LAYERS = [
    "threshold_layer.cpp.o",
    "tile_layer.cpp.o",
    "window_data_layer.cpp.o",
    "absval_layer.cpp.o",
    "accuracy_layer.cpp.o",
    "argmax_layer.cpp.o",
    "base_conv_layer.cpp.o",
    "base_data_layer.cpp.o",
    "batch_norm_layer.cpp.o",
    "batch_reindex_layer.cpp.o",
    "bias_layer.cpp.o",
    "bnll_layer.cpp.o",
    "concat_layer.cpp.o",
    "contrastive_loss_layer.cpp.o",
    "conv_layer.cpp.o",
    "crop_layer.cpp.o",
    "cudnn_conv_layer.cpp.o",
    "cudnn_lcn_layer.cpp.o",
    "cudnn_lrn_layer.cpp.o",
    "cudnn_pooling_layer.cpp.o",
    "cudnn_relu_layer.cpp.o",
    "cudnn_sigmoid_layer.cpp.o",
    "cudnn_softmax_layer.cpp.o",
    "cudnn_tanh_layer.cpp.o",
    "data_layer.cpp.o",
    "deconv_layer.cpp.o",
    "dropout_layer.cpp.o",
    "dummy_data_layer.cpp.o",
    "eltwise_layer.cpp.o",
    "elu_layer.cpp.o",
    "embed_layer.cpp.o",
    "euclidean_loss_layer.cpp.o",
    "exp_layer.cpp.o",
    "filter_layer.cpp.o",
    "flatten_layer.cpp.o",
    "hdf5_data_layer.cpp.o",
    "hdf5_output_layer.cpp.o",
    "hinge_loss_layer.cpp.o",
    "im2col_layer.cpp.o",
    "image_data_layer.cpp.o",
    "infogain_loss_layer.cpp.o",
    "inner_product_layer.cpp.o",
    "input_layer.cpp.o",
    "log_layer.cpp.o",
    "loss_layer.cpp.o",
    "lrn_layer.cpp.o",
    "memory_data_layer.cpp.o",
    "multinomial_logistic_loss_layer.cpp.o",
    "mvn_layer.cpp.o",
    "neuron_layer.cpp.o",
    "parameter_layer.cpp.o",
    "pooling_layer.cpp.o",
    "power_layer.cpp.o",
    "prelu_layer.cpp.o",
    "reduction_layer.cpp.o",
    "relu_layer.cpp.o",
    "reshape_layer.cpp.o",
    "scale_layer.cpp.o",
    "sigmoid_cross_entropy_loss_layer.cpp.o",
    "sigmoid_layer.cpp.o",
    "silence_layer.cpp.o",
    "slice_layer.cpp.o",
    "softmax_layer.cpp.o",
    "softmax_loss_layer.cpp.o",
    "split_layer.cpp.o",
    "spp_layer.cpp.o",
    "tanh_layer.cpp.o",
    "layer_factory.cpp.o",
]

genquery(
    name = "protobuf-root",
    expression = "@protobuf//:protobuf_lite",
    scope = ["@protobuf//:protobuf_lite"],
    opts = ["--output=location"]
)

genrule(
    name = "configure",
    message = "Building Caffe (this may take a while)",
    srcs = if_cuda([
        "@org_tensorflow//third_party/gpus/cuda:include/cudnn.h",
        "@org_tensorflow//third_party/gpus/cuda:" + cudnn_library_path()
    ]) + [
        ":protobuf-root", 
        "@protobuf//:protoc", 
        "@protobuf//:protobuf_lite"
    ],
    outs = [
        "lib/libcaffe.a", 
        "lib/libproto.a", 
        "include/caffe/proto/caffe.pb.h"
    ],
    cmd = 
        # build dir for cmake.
        # Adopt protobuf from bazel.
        '''
        srcdir=$$(pwd);
        workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); 

        protobuf_incl=$$(grep -oP "^/\\\S*(?=/)" $(location :protobuf-root))/src;
        protoc=$$srcdir/$(location @protobuf//:protoc);
        protolib=$$srcdir/$$(echo "$(locations @protobuf//:protobuf_lite)" | grep -o "\\\S*/libprotobuf_lite.a"); ''' + 

        # extra cmake options during cuda configuration,
        # adopting the tensorflow cuda configuration where
        # sensible.
        if_cuda(''' 
            cudnn_includes=$(location @org_tensorflow//third_party/gpus/cuda:include/cudnn.h);
            cudnn_lib=$(location @org_tensorflow//third_party/gpus/cuda:%s);
            extra_cmake_opts="-DCPU_ONLY:bool=OFF
                              -DUSE_CUDNN:bool=ON 
                              -DCUDNN_INCLUDE:path=$$srcdir/$$(dirname $$cudnn_includes)
                              -DCUDNN_LIBRARY:path=$$srcdir/$$cudnn_lib"; ''' % cudnn_library_path(), 
            '''extra_cmake_opts="-DCPU_ONLY:bool=ON";''') +

        # configure cmake.
        # openblas must be installed for this to 
        # succeed.
        '''
        pushd $$workdir;
        cmake $$srcdir/external/caffe_git         \
            -DCMAKE_INSTALL_PREFIX=$$srcdir/$(@D) \
            -DCMAKE_BUILD_TYPE=Release            \
            -DBLAS:string="open"                  \
            -DBUILD_python=OFF                    \
            -DBUILD_python_layer=OFF              \
            -DUSE_OPENCV=OFF                      \
            -DBUILD_SHARED_LIBS=OFF               \
            -DUSE_LEVELDB=OFF                     \
            -DUSE_LMDB=OFF                        \
            -DPROTOBUF_INCLUDE_DIR=$$protobuf_incl\
            -DPROTOBUF_PROTOC_EXECUTABLE=$$protoc \
            -DPROTOBUF_LIBRARY=$$protolib         \
            $${extra_cmake_opts}; ''' +

        # build libcaffe.a -- note we avoid building the 
        # caffe tools because 1) we don't need them anyway
        # and 2) they will fail to link because only 
        # protobuf_lite.a (and not libprotobuf.so) is
        # specified in PROTOBUF_LIBRARY.
        '''
        cmake --build . --target caffe -- -j 4;
        cp -r ./lib $$srcdir/$(@D)
        cp -r ./include $$srcdir/$(@D)
        
        popd;
        rm -rf $$workdir;''',
)

genrule(
    name = "cuda-extras",
    srcs = ["@org_tensorflow//third_party/gpus/cuda:cuda.config"],
    outs = ["lib64/libcurand.so." + cuda_sdk_version()],
    cmd  = '''
        source $(location @org_tensorflow//third_party/gpus/cuda:cuda.config) || exit -1;
        CUDA_TOOLKIT_PATH=$${CUDA_TOOLKIT_PATH:-/usr/local/cuda};
        FILE=libcurand.so.%s;
        SRC=$$CUDA_TOOLKIT_PATH/lib64/$$FILE;

        if test ! -e $$SRC; then
            echo "ERROR: $$SRC cannot be found";
            exit -1;
        fi

        mkdir -p $(@D);
        cp $$SRC $(@D)/$$FILE;''' % cuda_sdk_version(),
) 

# TODO(rayg): Bazel will ignore `alwayslink=1` for *.a archives (a bug?). 
#   This genrule unpacks the caffe.a so the object files can be linked 
#   independantly. A terrible hack, not least because we
#   need to know the layer names upfront.
genrule(
    name = "caffe-extract",
    srcs = [":configure", "lib/libcaffe.a"],
    outs = ["libcaffe.a.dir/" + o for o in CAFFE_WELL_KNOWN_LAYERS],
    cmd = '''
        workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); 
        cp $(location :lib/libcaffe.a) $$workdir; 
        pushd $$workdir; 
        ar x libcaffe.a; 
        popd;
        mkdir -p $(@D)/libcaffe.a.dir;
        cp -a $$workdir/*.o $(@D)/libcaffe.a.dir; 
        rm -rf $$workdir;
        ''',
)

cc_library(
    name = "curand",
    srcs = ["lib64/libcurand.so." + cuda_sdk_version()],
    data = ["lib64/libcurand.so." + cuda_sdk_version()],
    linkstatic = 1
)

cc_library(
    name = "caffe",
    srcs = [":caffe-extract", "lib/libcaffe.a", "lib/libproto.a"],
    hdrs = glob(["include/**"]) + ["include/caffe/proto/caffe.pb.h"],
    deps = if_cuda([
        "@org_tensorflow//third_party/gpus/cuda:cudnn", 
        "@org_tensorflow//third_party/gpus/cuda:cublas",
        ":curand",
    ]) + [
        "@protobuf//:protobuf"
    ],
    includes = ["include/"],
    defines = if_cuda([], ["CPU_ONLY"]),
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib",
        "-Wl,-rpath,/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib",
        "-lboost_system",
        "-lboost_thread",
        "-lboost_filesystem",
        "-lpthread",
        "-lglog",
        "-lgflags",
        "-lhdf5_hl",
        "-lhdf5",
        "-lz",
        "-ldl",
        "-lm",
        "-lopenblas",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    linkstatic = 1,
)
