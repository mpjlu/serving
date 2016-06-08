# Same as the TF if_cuda macro, but with a fully qualified
# reference to the cuda_crosstool_condition, so we can use it 
# in the caffe build.
def if_cuda(a, b=[]):
  return select({
      "@tf//third_party/gpus/cuda:cuda_crosstool_condition": a,
      "//conditions:default": b,
  })
