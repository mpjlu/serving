export CI_BUILD_PYTHON=python
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TENSORRT_INSTALL_PATH=/home/pmeng/TensorRT-4.0.1.6/lib
export TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1,7.0
export TF_CUDA_VERSION=9.0
export TF_CUDNN_VERSION=7
export TF_NCCL_VERSION=
export TF_TENSORRT_VERSION=4.1.2
export CUDNN_VERSION=7
export TMP=/tmp
bazel build --color=yes --curses=yes --verbose_failures --output_filter=DONT_MATCH_ANYTHING --config=nativeopt //tensorflow_serving/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package /tmp/pip && \
    pip --no-cache-dir install --upgrade /tmp/pip/tensorflow_serving*.whl
