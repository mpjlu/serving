load("@caffe_tools//:cuda.bzl", "if_cuda")
load("@org_tensorflow//third_party/gpus/cuda:platform.bzl",
     "cuda_sdk_version",
     "cudnn_library_path",
    )

package(default_visibility = ["//visibility:public"])

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
#   This genrule unpacks the caffe.a and merges the layers as a .o (ld -r).
#   (A terrible hack).
genrule(
    name = "caffe-extract",
    srcs = [":configure", "lib/libcaffe.a"],
    outs = ["libcaffe-layers.o"],
    cmd = '''
        workdir=$$(mktemp -d -t tmp.XXXXXXXXXX);
        cp $(location :lib/libcaffe.a) $$workdir;
        pushd $$workdir;
        ar x libcaffe.a;
        ld -r -o libcaffe-layers.o $$(echo layer_factory.cpp.o *_layer.*.o);
        popd;
        cp $$workdir/libcaffe-layers.o $(@D)/;
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
    defines = if_cuda(["USE_CUDNN"], ["CPU_ONLY"]),
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
