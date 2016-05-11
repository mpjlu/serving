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

#include <string>
#include <unordered_map>

#include "caffe/net.hpp"

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/caffe/caffe_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter for string-string hashmaps. It takes storage paths that give
// the locations of serialized hashmaps (in the format indicated in the config)
// and produces loaders for them.
class CaffeSourceAdapter
    : public SimpleLoaderSourceAdapter<StoragePath,
                                       caffe::Net<float>> {
 public:
  explicit CaffeSourceAdapter(const CaffeSourceAdapterConfig& config);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CaffeSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow
