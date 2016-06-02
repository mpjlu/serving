def if_cuda(a, b=[]):
  return select({
      "@tf//third_party/gpus/cuda:cuda_crosstool_condition": a,
      "//conditions:default": b,
  })
