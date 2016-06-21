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

load('//tensorflow_serving:workspace.bzl', 'tf_serving_workspace')
tf_serving_workspace()