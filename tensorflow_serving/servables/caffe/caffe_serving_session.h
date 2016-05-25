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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/core/status.h"

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace tensorflow {
namespace serving {

// Encapsulates a caffe network 
class CaffeServingSession {
 public:
  CaffeServingSession(const caffe::NetParameter& graph);
  virtual ~CaffeServingSession() = default;

  Status CopyTrainedLayersFromBinaryProto(const string trained_filename);

  virtual Status Run(const std::vector<std::pair<string, gtl::ArraySlice<float>>>& inputs,
                     const std::vector<string>& output_tensor_names,
                     std::vector<std::vector<float>>* outputs);

 private:
  std::unique_ptr<caffe::Net<float>> net_;
  std::unordered_map<string, unsigned int> input_blob_map_;
  std::unordered_map<string, unsigned int> output_blob_map_;
  std::mutex run_mutx_;

  TF_DISALLOW_COPY_AND_ASSIGN(CaffeServingSession);  
};

}  // namespace serving
}  // namespace tensorflow
