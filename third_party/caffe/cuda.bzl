# Same as the TF if_cuda macro, but with a fully qualified
# reference to the cuda_crosstool_condition, so we can use it 
# in the caffe build.
def if_cuda(if_true, if_false = []):
  return select({
      "@org_tensorflow//third_party/gpus/cuda:using_nvcc": if_true,
      "@org_tensorflow//third_party/gpus/cuda:using_gcudacc": if_true,
      "//conditions:default": if_false
  })
