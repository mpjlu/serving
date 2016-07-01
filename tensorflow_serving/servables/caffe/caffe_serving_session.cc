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

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
// note: this header resides in third_party/caffe
#include "openblas_prelude.h"

// avoid fp-16 redefinitions when using 
// a recent version of cuda
#if CUDA_VERSION >= 7050
#  define EIGEN_HAS_CUDA_FP16
#endif

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

// A guesstimate of the batch size; assume the outermost
// dimension of the input blob(s) indicates the batch size,
// unless the input is 1-dimensional, in which case assume
// batch size of 1. (I couldn't find much concrete 
// documentation on this.)
unsigned int BatchSizeOf(const caffe::Net<float>& net) {
  unsigned int x = 1;
  for (int idx : net.input_blob_indices()) {
    const std::vector<int>& shape = net.blobs().at(idx)->shape();
    if (shape.size() > 1 && shape[0] > 0) {
      x = std::max(x, (unsigned int)shape[0]);
    }
  }
  return x;
}

// Parse GPU ids or use all available devices
void GetGPUs(std::vector<int>* gpus) {
#ifndef CPU_ONLY
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; ++i) {
    if (caffe::Caffe::CheckDevice(i))
      gpus->push_back(i);
  }
#endif
}

bool TryAssignGPU() 
{
  std::vector<int> gpus;
  GetGPUs(&gpus);

  if (gpus.size() != 0) {
    caffe::Caffe::SetDevice(gpus[0]);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    return true;
  } 
  else {
    // tensorflow serving is a multi-threaded application;
    // avoid using multi-threaded OpenBLAS for now since this
    // is quite likely to cause problems when executing
    // caffe::forward(..) within a critical section 
    // (and could lead to deadlock).
    openblas_set_num_threads(1);
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    return false;
  }
}

CaffeServingSession::CaffeServingSession(const caffe::NetParameter& graph, 
                                         const CaffeSessionOptions& opts) 
    : net_{ nullptr }
    , batch_size_{ 0 }
    , ts_()
{
  bool is_gpu = ts_.run(
    [](std::unique_ptr<caffe::Net<float>>* net, 
       const caffe::NetParameter* graph) 
    {
      bool is_gpu = TryAssignGPU();
      net->reset(new caffe::Net<float>(*graph));
      return is_gpu;
    }, 
    &net_, 
    &graph);
  
  LOG(INFO) << "Caffe execution mode: " << 
    (is_gpu ? "GPU" : "CPU");
  {
    const std::vector<string>& blobs = net_->blob_names();
    for (int idx : net_->input_blob_indices()) {
      input_blob_map_.emplace(blobs[idx], idx);
    }
    for (int idx : net_->output_blob_indices()) {
      output_blob_map_.emplace(blobs[idx], idx);
    }
  }

  batch_size_ = BatchSizeOf(*net_);
  LOG(INFO) << "Loaded Network:"
      << "\n  name: " << net_->name() 
      << "\n  inputs: " << input_blob_map_.size() 
      << "\n  outputs: " << output_blob_map_.size() 
      << "\n  initial batch-size: " << batch_size_;
}

CaffeServingSession::~CaffeServingSession() = default;

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

  if (batch_size_ < batch_size) {
    TF_RETURN_IF_ERROR(Reshape(batch_size));
  }

  auto net_blobs = ts_.run(
    [](caffe::Net<float>* net) { return net->blobs(); }, 
    net_.get());

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

  ts_.run(
    [](caffe::Net<float>* net) { net->Forward(); }, 
    net_.get());

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

  return ts_.run(
    [](caffe::Net<float>* net, 
       const caffe::NetParameter& param) 
    {
      // TODO(rayg): this can abort
      net->CopyTrainedLayersFrom(param);
      return Status::OK();
    },
    net_.get(),
    param);
}

Status CaffeServingSession::Reshape(unsigned int batch_size)
{
  if (batch_size <= 0) { return errors::InvalidArgument("batch_size must be at least 1"); }
  if (batch_size_ == batch_size) { return Status::OK(); }

  batch_size_ = ts_.run(
    [](caffe::Net<float>* net,
       unsigned int batch_size) 
    {
      for (int idx : net->input_blob_indices()) {
        auto& blob = *(net->blobs().at(idx));
        std::vector<int> new_shape{ blob.shape() };
    
        if (new_shape.size() > 1 && new_shape[0] > 0) {
          new_shape[0] = batch_size;
          blob.Reshape(new_shape);
        }
      }
      net->Reshape();
      return batch_size;
    },
    net_.get(),
    batch_size);

  LOG(INFO) << "Reshaped Network (batch_size=" << batch_size_ << ").";
  return Status::OK();
}

} // namespace serving
} // namespace tensorflow
