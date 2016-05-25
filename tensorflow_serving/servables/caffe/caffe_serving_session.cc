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
#include <thread>
#include <mutex>

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

Status CaffeServingSession::Run(const std::vector<std::pair<string, gtl::ArraySlice<float>>>& inputs,
                                const std::vector<string>& output_tensor_names,
                                std::vector<std::vector<float>>* outputs)
{
  // don't permit parallel inference
  std::lock_guard<std::mutex> lock(run_mutx_);

  // check inputs are present, assuming there are no duplicates
  if (input_blob_map_.size() != inputs.size()) {
    return errors::InvalidArgument("Expected ", input_blob_map_.size(), 
                                   " inputs, but got ", inputs.size(), ".");
  }

  // copy input to network blobs
  auto net_blobs = net_->blobs();
  for (const std::pair<string, gtl::ArraySlice<float>>& in: inputs) {
    auto it = input_blob_map_.find(in.first);
    if (it == input_blob_map_.end()) {
      return errors::InvalidArgument("Input blob ", in.first,
        " does not exist in the network.");
    }
    else {
      unsigned idx = it->second;
      const gtl::ArraySlice<float>& view = in.second;
      std::copy_n(view.data(), view.size(), net_blobs[idx]->mutable_cpu_data());
    }
  }
  
  // run the inference
  net_->Forward();

  // copy to output vectors
  outputs->clear();
  for (const string& out: output_tensor_names) {
    auto it = output_blob_map_.find(out);
    if (it == output_blob_map_.end()) {
      return errors::InvalidArgument("Specified output '", out, 
                                     "' does not exist.");
    }
    auto idx = it->second;
    const float* begin = net_blobs[idx]->cpu_data();
    const float* end = begin + net_blobs[idx]->channels();
    outputs->emplace_back(begin, end);
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


