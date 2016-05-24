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

#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/text_format.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp "

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

namespace tensorflow {
namespace serving {
namespace {

// Create a network using the given options and load the graph.
Status CreateSessionFromGraphDef(
    const CaffeSessionOptions& options, 
    const caffe::NetParameter& graph,
    std::unique_ptr<CaffeServingSession>* session) {
  // TODO(rayglover): is there any way to handle failure here?
  session->reset(new CaffeServingSession(graph));
  return Status::OK();
}

Status GetGraphDefFromExport(const StringPiece export_dir,
                             caffe::NetParameter* graph_def) {
  const string graph_def_path =
      tensorflow::io::JoinPath(export_dir, kGraphDefFilename);
  
  if (!Env::Default()->FileExists(graph_def_path)) {
    return errors::NotFound(
        strings::StrCat("Caffe model does not exist: ",
                        graph_def_path));
  }
  else if (!ReadProtoFromTextFile(graph_def_path, graph_def)) {
    return errors::InvalidArgument(
      strings::StrCat("Caffe network failed to load from file: ",
                      graph_def_path));
  } else if (!UpgradeNetAsNeeded(graph_def_path, graph_def)) {
    return errors::InvalidArgument(
      strings::StrCat("Network upgrade failed from while loading from file: ",
                      graph_def_path));
  }
  return Status::OK();
}

string GetVariablesFilename(const StringPiece export_dir) {
  const char kVariablesFilename[] = "weights.caffemodel";
  return tensorflow::io::JoinPath(export_dir, kVariablesFilename);
}

Status RunRestoreOp(const StringPiece export_dir,
                    CaffeServingSession* session) {
  LOG(INFO) << "Running restore op for CaffeSessionBundle";
  string weights_path = GetVariablesFilename(export_dir);
  if (Env::Default()->FileExists(weights_path)) {
    return session->CopyTrainedLayersFromBinaryProto(weights_path);
  } else {
    return errors::NotFound(
        strings::StrCat("Caffe weights file does not exist: ",
                        weights_path));
  }
}

}  // namespace

tensorflow::Status LoadSessionBundleFromPath(
    const CaffeSessionOptions& options, 
    const StringPiece export_dir,
    CaffeSessionBundle* bundle) 
{
  LOG(INFO) << "Attempting to load a SessionBundle from: " << export_dir;
  // load prototxt
  TF_RETURN_IF_ERROR(
      GetGraphDefFromExport(export_dir, &(bundle->graph_def)));

  // initialize network
  const caffe::NetParameter& graph_def = bundle->graph_def;
  TF_RETURN_IF_ERROR(
      CreateSessionFromGraphDef(options, graph_def, &bundle->session));

  // load weights
  TF_RETURN_IF_ERROR(
      RunRestoreOp(export_dir, bundle->session.get()));

  LOG(INFO) << "Done loading SessionBundle";
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
