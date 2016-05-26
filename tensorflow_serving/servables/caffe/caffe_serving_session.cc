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
#include <unordered_map>

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

// Constructs a flat tensor with 'vals'.
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals) {
  Tensor ret(DataTypeToEnum<T>::value, {static_cast<int64>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

CaffeServingSession::CaffeServingSession(const caffe::NetParameter& graph) 
    : net_{ new caffe::Net<float>(graph) } 
{
  std::vector<string> blobs = net_->blob_names();

  for (int idx : net_->input_blob_indices()) {
    input_blob_map_.emplace(blobs[idx], idx);
  }
  for (int idx : net_->output_blob_indices()) {
    output_blob_map_.emplace(blobs[idx], idx);
  }

  LOG(INFO) << "Network has " << input_blob_map_.size() 
      << " inputs and " << output_blob_map_.size() << " outputs";
}

Status CaffeServingSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                                const std::vector<string>& output_tensor_names,
                                const std::vector<string>& target_node_names,
                                std::vector<Tensor>* outputs)
{
  // can't do anything with target_nodes..
  if (target_node_names.size() > 0) {
    return errors::InvalidArgument("target_node_names is not supported by ",
                                   "the Caffe backend");
  }

  // check inputs are present, assuming there are no duplicates
  if (inputs.size() == 0 || inputs.size() < input_blob_map_.size()) {
    return errors::InvalidArgument("Expected ", input_blob_map_.size(), 
                                   " inputs, but got ", inputs.size(), ".");
  }

  // determine the batch size from the first input only
  unsigned int batch_size = 0;
  {
    const Tensor& in = inputs[0].second;
    if (in.dims() < 2) {
      return errors::InvalidArgument("Could not determine the batch size; "
                                     "input must have at least 2 dimensions");
    }
    batch_size = in.dim_size(0);
    if (batch_size < 1) {
      return errors::InvalidArgument("Invalid batch size of ", batch_size); 
    }
  }

  // copy input to network blobs, validating tensor dimensions, etc.
  auto net_blobs = net_->blobs();
  for (const std::pair<string, Tensor>& in: inputs) {
    auto it = input_blob_map_.find(in.first);
    if (it == input_blob_map_.end()) {
      return errors::InvalidArgument("Input Tensor ", in.first,
        " does not exist in the network.");
    }
    else {
      if (in.second.dim_size(0) != batch_size) {
        return errors::InvalidArgument("Input Tensor ", in.first,
        " has an incorrect batch size.");
      }
      // TODO(rayg): validate all other dimensions before copy
      const auto view = in.second.flat<float>();
      unsigned idx = it->second;
      std::copy_n(view.data(), view.size(), net_blobs[idx]->mutable_cpu_data());
    }
  }

//  LOG(INFO) << ">> BATCH " << batch_size;
  
  // run the inference
  net_->Forward();

  // copy to output vectors
  outputs->clear();
  for (const string& out: output_tensor_names) {
    auto it = output_blob_map_.find(out);
    if (it == output_blob_map_.end()) {
      return errors::InvalidArgument("Specified network output '", out, 
                                     "' does not exist.");
    }
    caffe::Blob<float>& blob = *net_blobs[it->second];
    // 2-D output
    {
      TensorShape shape{ batch_size, blob.channels() };
      Tensor t = AsTensor<float>({ blob.cpu_data(), batch_size * (unsigned long)blob.channels() }, shape);
      outputs->push_back(t);
    }
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
  // TODO(rayg): this can abort
  net_->CopyTrainedLayersFrom(param);
  return Status::OK();
}

} // namespace serving
} // namespace tensorflow


