# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

# All TensorFlow Serving external dependencies.
# workspace_dir is the absolute path to the TensorFlow Serving repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/serving"'
def tf_serving_workspace(workspace_dir):
  native.local_repository(
    name = "org_tensorflow",
    path = workspace_dir + "/tensorflow",
  )

  native.local_repository(
    name = "inception_model",
    path = workspace_dir + "/tf_models/inception",
  )

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )

  # ===== caffe dependencies =====
  native.bind(
    name = "caffe",
    actual = "@caffe_git//:caffe",
  )

  native.new_git_repository(
    name = "caffe_git",
    remote = "https://github.com/BVLC/caffe",
    commit = "50c9a0fc8eed0101657e9f8da164d88f66242aeb",
    init_submodules = True,
    build_file = workspace_dir + "/caffe.BUILD",
  )

  native.local_repository(
    name = "caffe_tools",
    path = workspace_dir + "/third_party/caffe"
  )
