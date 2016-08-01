workspace(name = "tf_serving")

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load('//tensorflow_serving:workspace.bzl', 'tf_serving_workspace')
tf_serving_workspace(__workspace_dir__)

# Please add all new TensorFlow dependencies in tensorflow/workspace.bzl.
load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace(__workspace_dir__ + "/tensorflow/", "@org_tensorflow")
