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
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp "

namespace tensorflow {
namespace serving {
namespace {

// Create a network using the given options and load the graph.
Status CreateSessionFromGraphDef(
    const CaffeSessionOptions& options, 
    const caffe::NetParameter& graph,
    std::unique_ptr<caffe::Net<float>>* session) {
  // TODO(rayglover): is there any way to handle failure here?
  session->reset(new caffe::Net<float>(graph));
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
  else if (ReadProtoFromTextFile(graph_def_path, graph_def)) {
    if (UpgradeNetAsNeeded(graph_def_path, graph_def)) {
      return Status::OK();
    }
  }
  return errors::FailedPrecondition(
    strings::StrCat("Caffe network failed to load from file: ",
                    graph_def_path));
}

string GetVariablesFilename(const StringPiece export_dir) {
  const char kVariablesFilename[] = "weights.caffemodel";
  return tensorflow::io::JoinPath(export_dir, kVariablesFilename);
}

Status RunRestoreOp(const StringPiece export_dir,
                    caffe::Net<float>* session) {
  LOG(INFO) << "Running restore op for CaffeSessionBundle";
  string weights_path = GetVariablesFilename(export_dir);
  if (Env::Default()->FileExists(weights_path)) {
    session->CopyTrainedLayersFromBinaryProto(weights_path);
    return Status::OK();  
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

//auto collection_def = bundle->meta_graph_def.collection_def();
//if (collection_def.find(kGraphKey) != collection_def.end()) {
//  // Use serving graph_def in MetaGraphDef collection_def.
//  if (collection_def[kGraphKey].any_list().value_size() != 1) {
//    return errors::FailedPrecondition(
//        strings::StrCat("Expected exactly one serving GraphDef in : ",
//                        bundle->meta_graph_def.DebugString()));
//  }
//  tensorflow::GraphDef graph_def;
//  collection_def[kGraphKey].any_list().value(0).UnpackTo(&graph_def);
//  TF_RETURN_IF_ERROR(
//      CreateSessionFromGraphDef(options, graph_def, &bundle->session));
//} else {
    // Fallback to use the graph_def in the MetaGraphDef.
  // initialize network
  const caffe::NetParameter& graph_def = bundle->graph_def;
  TF_RETURN_IF_ERROR(
      CreateSessionFromGraphDef(options, graph_def, &bundle->session));
//}

//  std::vector<AssetFile> asset_files;
//  auto any_assets = collection_def[kAssetsKey].any_list().value();
//  for (const auto any_asset : any_assets) {
//    AssetFile asset_file;
//    any_asset.UnpackTo(&asset_file);
//    asset_files.push_back(asset_file);
//  }
//
  // load weights
  TF_RETURN_IF_ERROR(
      RunRestoreOp(export_dir, bundle->session.get()));
//
//  if (collection_def.find(kInitOpKey) != collection_def.end()) {
//    if (collection_def[kInitOpKey].node_list().value_size() != 1) {
//      return errors::FailedPrecondition(
//          strings::StrCat("Expected exactly one serving init op in : ",
//                          bundle->meta_graph_def.DebugString()));
//    }
//    return RunInitOp(export_dir, asset_files,
//                     collection_def[kInitOpKey].node_list().value(0),
//                     bundle->session.get());
//}

  LOG(INFO) << "Done loading SessionBundle";
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
