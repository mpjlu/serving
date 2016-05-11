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

#include "tensorflow_serving/servables/caffe/caffe_source_adapter.h"

#include <stddef.h>
#include <memory>
#include <vector>

#include "caffe/net.hpp"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"
//#include "tensorflow/core/platform/env.h"
//#include "tensorflow/core/platform/types.h

namespace tensorflow {
namespace serving {
namespace {

using Caffe = caffe::Net<float>;

// Populates a caffe from a file located at 'path', in format 'format'.
Status LoadCaffeFromFile(const string& path,
                           const CaffeSourceAdapterConfig::Format& format,
                           std::unique_ptr<Caffe>* caffe) {
  caffe->reset(new Caffe(path, caffe::TEST));
  return Status::OK();
}
}  // namespace

CaffeSourceAdapter::CaffeSourceAdapter(
    const CaffeSourceAdapterConfig& config)
  : SimpleLoaderSourceAdapter<StoragePath, Caffe>(
          [config](const StoragePath& path, std::unique_ptr<Caffe>* caffe) {
            return LoadCaffeFromFile(path, config.format(), caffe);
          },
          // Decline to supply a resource footprint estimate.
          SimpleLoaderSourceAdapter<StoragePath,
                                    Caffe>::EstimateNoResources()) {}

}  // namespace serving
}  // namespace tensorflow
