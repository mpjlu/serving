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

// gRPC server implementation of
// tensorflow_serving/apis/prediction_service.proto.
//
// It bring up a standard server to serve a single TensorFlow model using
// command line flags, or multiple models via config file.
//
// ModelServer prioritizes easy invocation over flexibility,
// and thus serves a statically configured set of models. New versions of these
// models will be loaded and managed over time using the EagerLoadPolicy at:
//     tensorflow_serving/core/eager_load_policy.h.
// by AspiredVersionsManager at:
//     tensorflow_serving/core/aspired_versions_manager.h
//
// ModelServer has inter-request batching support built-in, by using the
// BatchingSession at:
//     tensorflow_serving/batching/batching_session.h
//
// To serve a single model, run with:
//     $path_to_binary/tensorflow_model_server \
//     --model_base_path=[/tmp/my_model | gs://gcs_address] \
// To specify model name (default "default"): --model_name=my_name
// To specify port (default 8500): --port=my_port
// To enable batching (default disabled): --enable_batching
// To log on stderr (default disabled): --alsologtostderr

#include <unistd.h>
#include <iostream>
#include <memory>
#include <utility>

#include "google/protobuf/wrappers.pb.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/apis/prediction_service.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/model_servers/server_core.h"

#if USE_TENSORFLOW
 #include "tensorflow/core/platform/init_main.h"
 #include "tensorflow_serving/servables/tensorflow/predict_impl.h"
 #include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"
#endif

#if USE_CAFFE
 #include "tensorflow_serving/servables/caffe/predict_impl.h"
 #include "tensorflow_serving/servables/caffe/caffe_source_adapter.h"
#endif

using tensorflow::serving::BatchingParameters;
using tensorflow::serving::EventBus;
using tensorflow::serving::Loader;
using tensorflow::serving::ModelServerConfig;
using tensorflow::serving::ServableState;
using tensorflow::serving::ServableStateMonitor;
using tensorflow::serving::ServerCore;
using tensorflow::serving::Target;
using tensorflow::serving::UniquePtrWithDeps;
using tensorflow::string;

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

namespace {

struct TensorFlow {};
struct Caffe {};

template<typename>
struct ServableTraits {
  static const char* name() { return ""; }
  static constexpr bool available = false;
  static void GlobalInit(int argc, char** argv) {}
};

#if USE_TENSORFLOW
template <>
struct ServableTraits<TensorFlow> {
  static const char* name() { return "tensorflow"; }
  static constexpr bool available = true;

  using SourceAdapterConfig = tensorflow::serving::SessionBundleSourceAdapterConfig;
  using SourceAdapter       = tensorflow::serving::SessionBundleSourceAdapter;
  using PredictImpl         = tensorflow::serving::TensorflowPredictImpl;

  static void GlobalInit(int argc, char** argv) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);
  }
};
#endif

#if USE_CAFFE
template <>
struct ServableTraits<Caffe> {
  static const char* name() { return "caffe"; }
  static constexpr bool available = true;

  using SourceAdapterConfig = tensorflow::serving::CaffeSourceAdapterConfig;
  using SourceAdapter       = tensorflow::serving::CaffeSourceAdapter;
  using PredictImpl         = tensorflow::serving::CaffePredictImpl;

  static void GlobalInit(int argc, char** argv) {
    tensorflow::serving::CaffeGlobalInit(&argc, &argv);
  }
};
#endif

template<typename S>
tensorflow::Status CreateSourceAdapter(
    const typename ServableTraits<S>::SourceAdapterConfig& config,
    const string& model_type,
    std::unique_ptr<ServerCore::ModelServerSourceAdapter>* adapter) {
  using T = ServableTraits<S>;
  CHECK(model_type == T::name())  // Crash ok
      << "ModelServer supports only " << T::name() << " model.";
  std::unique_ptr<typename T::SourceAdapter> typed_adapter;
  TF_RETURN_IF_ERROR(T::SourceAdapter::Create(config, &typed_adapter));
  *adapter = std::move(typed_adapter);
  return tensorflow::Status::OK();
}

tensorflow::Status CreateServableStateMonitor(
    EventBus<ServableState>* event_bus,
    std::unique_ptr<ServableStateMonitor>* monitor) {
  monitor->reset(new ServableStateMonitor(event_bus));
  return tensorflow::Status::OK();
}

tensorflow::Status LoadDynamicModelConfig(
    const ::google::protobuf::Any& any,
    Target<std::unique_ptr<Loader>>* target) {
  CHECK(false)  // Crash ok
      << "ModelServer does not yet support dynamic model config.";
}

template<typename S>
ModelServerConfig BuildSingleModelConfig(const string& model_name,
                                         const string& model_base_path) {
  using T = ServableTraits<S>;
  ModelServerConfig config;
  LOG(INFO) << "Building single " << T::name() << " model file config: "
            << " model_name: " << model_name
            << " model_base_path: " << model_base_path;
  tensorflow::serving::ModelConfig* single_model =
      config.mutable_model_config_list()->add_config();
  single_model->set_name(model_name);
  single_model->set_base_path(model_base_path);
  single_model->set_model_type(T::name());
  return config;
}

grpc::Status ToGRPCStatus(const tensorflow::Status& status) {
  const int kErrorMessageLimit = 1024;
  string error_message;
  if (status.error_message().length() > kErrorMessageLimit) {
    LOG(ERROR) << "Truncating error: " << status.error_message();
    error_message =
        status.error_message().substr(0, kErrorMessageLimit) + "...TRUNCATED";
  } else {
    error_message = status.error_message();
  }
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      error_message);
}

template<typename S>
class PredictionServiceImpl final : public PredictionService::Service {
 public:

  explicit PredictionServiceImpl(std::unique_ptr<ServerCore> core)
      : core_(std::move(core)) {}

  grpc::Status Predict(ServerContext* context, const PredictRequest* request,
                       PredictResponse* response) override {
    return ToGRPCStatus(ServableTraits<S>::PredictImpl::Predict(
        core_.get(), *request, response));
  }

 private:
  std::unique_ptr<ServerCore> core_;
};

template<typename S>
void RunServer(int port, std::unique_ptr<ServerCore> core) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);
  PredictionServiceImpl<S> service(std::move(core));
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running ModelServer at " << server_address << " ...";
  server->Wait();
}

template<typename S,
    typename std::enable_if< ServableTraits<S>::available, int >::type = 0>
int BuildAndRun(tensorflow::int32 port,
                bool enable_batching,
                tensorflow::string model_name,
                tensorflow::string model_base_path) {
  ModelServerConfig config =
      BuildSingleModelConfig<S>(model_name, model_base_path);

  typename ServableTraits<S>::SourceAdapterConfig source_adapter_config;
  // Batching config
  if (enable_batching) {
    BatchingParameters* batching_parameters =
        source_adapter_config.mutable_config()->mutable_batching_parameters();
    batching_parameters->mutable_thread_pool_name()->set_value(
        "model_server_batch_threads");
  }

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(
      config, std::bind(CreateSourceAdapter<S>, source_adapter_config,
                        std::placeholders::_1, std::placeholders::_2),
      &CreateServableStateMonitor, &LoadDynamicModelConfig, &core));
  RunServer<S>(port, std::move(core));
  return 0;
}

template<typename S,
    typename std::enable_if< !ServableTraits<S>::available, int >::type = 0>
int BuildAndRun(
    tensorflow::int32, bool, tensorflow::string, tensorflow::string) {
  std::cout << "Servable type unavailable. Did you compile it?" << std::endl;
  return -1;
}

} // namespace

int main(int argc, char** argv) {
  tensorflow::int32 port = 8500;
  bool enable_batching = false;
  tensorflow::string model_name = "default";
  tensorflow::string servable = ServableTraits<TensorFlow>::name();
  tensorflow::string model_base_path;
  const bool parse_result = tensorflow::ParseFlags(
      &argc, argv, {tensorflow::Flag("port", &port),
                    tensorflow::Flag("enable_batching", &enable_batching),
                    tensorflow::Flag("model_name", &model_name),
                    tensorflow::Flag("servable", &servable),
                    tensorflow::Flag("model_base_path", &model_base_path)});
  if (!parse_result || model_base_path.empty()) {
    std::cout << "Usage: model_server"
              << " [--port=8500]"
              << " [--enable_batching]"
              << " [--model_name=my_name]"
              << " [--servable=tensorflow]"
              << " --model_base_path=/path/to/export" << std::endl;
    return -1;
  }

  if (servable == ServableTraits<Caffe>::name()) {
    ServableTraits<Caffe>::GlobalInit(argc, argv);
    return BuildAndRun<Caffe>(port, enable_batching, model_name, model_base_path);
  }
  else if (servable == ServableTraits<TensorFlow>::name()) {
    ServableTraits<TensorFlow>::GlobalInit(argc, argv);
    return BuildAndRun<TensorFlow>(port, enable_batching, model_name, model_base_path);
  }
  else {
    std::cout << "Invalid servable name." << std::endl;
    return -1;
  }
}
