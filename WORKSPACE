workspace(name = "tf_serving")

local_repository(
  name = "org_tensorflow",
  path = __workspace_dir__ + "/tensorflow",
)

local_repository(
  name = "inception_model",
  path = __workspace_dir__ + "/tf_models/inception",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/", "@org_tensorflow")

# ===== gRPC dependencies =====

bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

# ===== Caffe =====

bind(
    name = "caffe",
    actual = "@caffe_git//:caffe",
)

new_git_repository(
    name = "caffe_git",
    remote = "https://github.com/BVLC/caffe",
    commit = "a934ca54f3633479ea0573346c510df4f757df6c",
    init_submodules = True,
    build_file = "caffe.BUILD",
)
