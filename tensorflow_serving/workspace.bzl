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
    commit = "a934ca54f3633479ea0573346c510df4f757df6c",
    init_submodules = True,
    build_file = path_prefix + "caffe.BUILD",
  )

  native.local_repository(
    name = "caffe_tools",
    path = "third_party/caffe"
  )
