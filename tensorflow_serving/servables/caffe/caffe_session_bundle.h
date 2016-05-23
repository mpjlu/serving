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

// Low-level functionality for setting up a inference Session.

#pragma once

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace tensorflow {
namespace serving {

const char kGraphDefFilename[] = "deploy.prototxt";

// No session options for Caffe
struct CaffeSessionOptions {};

// The closest thing we can get to a TF session bundle?
struct CaffeSessionBundle {
  std::unique_ptr<caffe::Net<float>> session;
  caffe::NetParameter graph_def;
};

// Loads a manifest and initialized session using the output of an Exporter
tensorflow::Status LoadSessionBundleFromPath(
    const CaffeSessionOptions& options,
    const tensorflow::StringPiece export_dir, 
    CaffeSessionBundle* bundle);

}  // namespace serving
}  // namespace tensorflow
