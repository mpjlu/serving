/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace tensorflow {
namespace serving {

Status CaffeServingSession::Run(const std::vector<gtl::ArraySlice<float>>& inputs,
                                std::vector<std::vector<float>>& outputs)
{
  const std::vector<caffe::Blob<float>*>& net_inputs = net_->input_blobs();
  const std::vector<caffe::Blob<float>*>& net_outputs = net_->output_blobs();

  if (net_inputs.size() != inputs.size()) {
    return errors::InvalidArgument("expected ", net_inputs.size(), 
                                   " inputs, but got ", inputs.size(), ".");
  }

  for (unsigned i=0; i < inputs.size(); ++i) {
    if (net_inputs[i]->count() < (signed)inputs[i].size()) {
      return errors::InvalidArgument("input ", i, " has invalid dimension of ", inputs[i].size());
    }
    std::copy_n(inputs[i].data(), inputs[i].size(), net_inputs[i]->mutable_cpu_data());
  }

  // run the inference
  net_->Forward();

  // copy to output vectors
  outputs.clear();

  for (unsigned i=0; i < net_outputs.size(); ++i) {
    // TODO(rayg): this makes some fairly big assumptions about the output shape
    const float* begin = net_outputs[i]->cpu_data();
    const float* end = begin + net_outputs[i]->channels();

    outputs.emplace_back(begin, end);
  }
  return Status::OK();
}

Status CaffeServingSession::CopyTrainedLayersFromBinaryProto(const string trained_filename)
{
  caffe::NetParameter param;

  if (!caffe::ReadProtoFromBinaryFile(trained_filename, &param)) {
    return errors::InvalidArgument(
      strings::StrCat("Caffe network failed to load pretrained layers from file: ",
                      trained_filename));  
  }
  // note: this can fail
  net_->CopyTrainedLayersFrom(param);
  return Status::OK();
}

} // namespace serving
} // namespace tensorflow