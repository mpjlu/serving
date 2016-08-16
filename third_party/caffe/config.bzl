def if_pycaffe(if_true, if_false = []):
  return select({
      "@caffe_tools//:caffe_python_layer": if_true,
      "//conditions:default": if_false
  })
