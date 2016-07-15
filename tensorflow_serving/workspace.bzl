# TensorFlow serving external dependencies that can be loaded in WORKSPACE files.

def tf_serving_workspace(path_prefix = ""):
  native.bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
  )

  native.bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
  )

  native.bind(
    name = "caffe",
    actual = "@caffe_git//:caffe",
  )

  native.new_git_repository(
    name = "caffe_git",
    remote = "https://github.com/BVLC/caffe",
    commit = "50c9a0fc8eed0101657e9f8da164d88f66242aeb",
    init_submodules = True,
    build_file = path_prefix + "caffe.BUILD",
  )

  native.local_repository(
    name = "caffe_tools",
    path = path_prefix + "third_party/caffe"
  )
