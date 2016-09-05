/* Copyright 2016 IBM Corp. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================ */

// A gRPC server that classifies objects within images.

#include <stddef.h>
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// gRPC
#include "grpc++/completion_queue.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/async_unary_call.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"

// Tensor + utilities
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

// TFS
#include "tensorflow_serving/batching/basic_batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/session_bundle/signature.h"

// service api
#include "tensorflow_serving/example/rcnn_detector.grpc.pb.h"
#include "tensorflow_serving/example/rcnn_detector.pb.h"

// caffe servable
#include "tensorflow_serving/servables/caffe/caffe_simple_servers.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_signature.h"

// rcnn utils
#include "tensorflow_serving/example/rcnn_utils.h"

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using grpc::StatusCode;

using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::serving::ClassificationSignature;

using tensorflow::serving::DetectRequest;
using tensorflow::serving::DetectResponse;
using tensorflow::serving::DetectService;
using tensorflow::serving::Detection;

namespace {
const int kImageSizeW = 800;
const int kImageSizeH = 600;
const int kNumChannels = 3;
const int kImageDataSize = kImageSizeW * kImageSizeH * kNumChannels;

class DetectServiceImpl;

// Class encompassing the state and logic needed to serve a request.
class CallData {
 public:
  CallData(DetectServiceImpl* service_impl,
           DetectService::AsyncService* service,
           ServerCompletionQueue* cq);

  void Proceed();

  void Finish(Status status);

  const DetectRequest& request() { return request_; }

  DetectResponse* mutable_response() { return &response_; }

 private:
  // Service implementation.
  DetectServiceImpl* service_impl_;

  // The means of communication with the gRPC runtime for an asynchronous
  // server.
  DetectService::AsyncService* service_;
  // The producer-consumer queue where for asynchronous server notifications.
  ServerCompletionQueue* cq_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  DetectRequest request_;
  // What we send back to the client.
  DetectResponse response_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<DetectResponse> responder_;

  // Let's implement a tiny state machine with the following states.
  enum CallStatus { CREATE, PROCESS, FINISH };
  CallStatus status_;  // The current serving state.
};

// A Task holds all of the information for a single inference request.
struct Task : public tensorflow::serving::BatchTask {
  ~Task() override = default;
  size_t size() const override { return 1; }

  Task(CallData* calldata_arg)
      : calldata(calldata_arg) {}

  CallData* calldata;
};

class DetectServiceImpl final {
 public:
  DetectServiceImpl(const string& servable_name,
                   std::unique_ptr<tensorflow::serving::Manager> manager);

  void Detect(CallData* call_data);

  // Produces classifications for a batch of requests and associated responses.
  void DoDetectInBatch(
      std::unique_ptr<tensorflow::serving::Batch<Task>> batch);

  // Name of the servable to use for inference.
  const string servable_name_;
  // Manager in charge of loading and unloading servables.
  std::unique_ptr<tensorflow::serving::Manager> manager_;
  // A scheduler for batching multiple request calls into single calls to
  // Session->Run().
  std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>
      batch_scheduler_;
};

// Take in the "service" instance (in this case representing an asynchronous
// server) and the completion queue "cq" used for asynchronous communication
// with the gRPC runtime.
CallData::CallData(DetectServiceImpl* service_impl,
                   DetectService::AsyncService* service,
                   ServerCompletionQueue* cq)
    : service_impl_(service_impl),
      service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
  // Invoke the serving logic right away.
  Proceed();
}

void CallData::Proceed() {
  if (status_ == CREATE) {
    // As part of the initial CREATE state, we *request* that the system
    // start processing Detect requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    service_->RequestDetect(&ctx_, &request_, &responder_, cq_, cq_, this);
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new CallData(service_impl_, service_, cq_);
    // Start processing.
    service_impl_->Detect(this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (CallData).
    delete this;
  }
}

void CallData::Finish(Status status) {
  status_ = FINISH;
  responder_.Finish(response_, status, this);
}

DetectServiceImpl::DetectServiceImpl(
    const string& servable_name,
    std::unique_ptr<tensorflow::serving::Manager> manager)
  : servable_name_(servable_name)
  , manager_(std::move(manager)) 
{
  tensorflow::serving::BasicBatchScheduler<Task>::Options scheduler_options;
  scheduler_options.thread_pool_name = "rcnn_service_batch_threads";
  // Use a large queue, to avoid rejecting requests. (Note: a production
  // server with load balancing may want to use the default, much smaller,
  // value.)
  scheduler_options.max_enqueued_batches = 250;
  // py-faster-rcnn doesnt support minibatches larger than 1.
  scheduler_options.max_batch_size = 1;
  TF_CHECK_OK(tensorflow::serving::BasicBatchScheduler<Task>::Create(
      scheduler_options,
      [this](std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
        this->DoDetectInBatch(std::move(batch));
      },
      &batch_scheduler_));
}

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

// WARNING(break-tutorial-inline-code): The following code snippet is
// in-lined in tutorials, please update tutorial documents accordingly
// whenever code changes.
void DetectServiceImpl::Detect(CallData* calldata) {
  // Verify input.
  if (calldata->request().image_data().size() != kImageDataSize) {
    calldata->Finish(
        Status(StatusCode::INVALID_ARGUMENT,
               tensorflow::strings::StrCat(
                   "expected image_data of size ", kImageDataSize,
                   ", got ", calldata->request().image_data().size())));
    return;
  }

  // Create and submit a task to the batch scheduler.
  std::unique_ptr<Task> task(new Task(calldata));
  tensorflow::Status status = batch_scheduler_->Schedule(&task);

  if (!status.ok()) {
    calldata->Finish(ToGRPCStatus(status));
    return;
  }
}

// Produces classifications for a batch of requests and associated responses.
void DetectServiceImpl::DoDetectInBatch(
    std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
  batch->WaitUntilClosed();
  if (batch->empty()) {
    return;
  }

  const int batch_size = batch->num_tasks();
  // Replies to each task with the given error status.
  auto complete_with_error = [&batch](StatusCode code, const string& msg) {
    Status status(code, msg);
    for (int i = 0; i < batch->num_tasks(); i++) {
      Task* task = batch->mutable_task(i);
      task->calldata->Finish(status);
    }
  };
  if (batch_size > 1) {
    // currently not supported by py-faster-rcnn
    complete_with_error(StatusCode::INTERNAL, "batch size > 1");
    return;
  }

  // Get a handle to the SessionBundle.  The handle ensures the Manager does
  // not reload this while it is in use.
  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  auto handle_request =
      tensorflow::serving::ServableRequest::Latest(servable_name_);
  tensorflow::serving::ServableHandle<tensorflow::serving::CaffeSessionBundle>
      bundle;
  const tensorflow::Status lookup_status =
      manager_->GetServableHandle(handle_request, &bundle);

  if (!lookup_status.ok()) {
    complete_with_error(StatusCode::INTERNAL,
                        lookup_status.error_message());
    return;
  }

  // Transform protobuf input to inference input tensor.
  // See mnist_model.py for details.
  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  Tensor im_blob(tensorflow::DT_FLOAT, {batch_size, kImageDataSize});
  {
    auto dst = im_blob.flat_outer_dims<float>().data();
    for (int i = 0; i < batch_size; ++i) {
      const auto& im = batch->mutable_task(i)->calldata->request().image_data();
      std::transform(im.begin(), im.end(), dst,
          [](const uint8_t& a) { return static_cast<float>(a); });
      dst += kImageDataSize;
    }
  }
  // BGR means-subtraction
  tensorflow::Status means_status =
    rcnn::BatchBGRMeansSubtract(&im_blob);

  if (!means_status.ok()) {
    complete_with_error(StatusCode::INTERNAL, means_status.error_message());
    return;
  }

  Tensor im_info(tensorflow::DT_FLOAT, {batch_size, 3});
  {
    auto dst = im_info.flat<float>().data();
    dst[0] = kImageSizeH;
    dst[1] = kImageSizeW;
    dst[2] = 1.0 /* scale */;
  }

  tensorflow::Tensor scores;
  tensorflow::Tensor boxes;
  tensorflow::Tensor class_labels;

  // Run classification.
  tensorflow::Status run_status =
    rcnn::RunClassification(
        im_blob, im_info, bundle->session.get(),
        &boxes, &scores, &class_labels);

  if (!run_status.ok()) {
    complete_with_error(StatusCode::INTERNAL, run_status.error_message());
    return;
  }

  std::vector<rcnn::Detection> dets;
  // Post-process bounding boxes
  run_status = rcnn::ProcessDetections(
      &boxes, &scores, &dets);

  // Transform inference output tensor to protobuf output.
  auto labels_mat = class_labels.flat<string>();
  for (int i = 0; i < batch_size; ++i) {
    auto calldata = batch->mutable_task(i)->calldata;
    DetectResponse* resp = calldata->mutable_response();

    for (const auto& det : dets) {
      if (det.class_idx == 0)
        continue;
      Detection* det_proto = resp->add_detections();
      det_proto->set_roi_x1(det.roi_rect[0]);
      det_proto->set_roi_y1(det.roi_rect[1]);
      det_proto->set_roi_x2(det.roi_rect[2]);
      det_proto->set_roi_y2(det.roi_rect[3]);
      det_proto->set_score(det.score);
      det_proto->set_class_label(labels_mat(det.class_idx));
    }
    calldata->Finish(Status::OK);
  }
}

void HandleRpcs(DetectServiceImpl* service_impl,
                DetectService::AsyncService* service,
                ServerCompletionQueue* cq) {
  // Spawn a new CallData instance to serve new clients.
  new CallData(service_impl, service, cq);
  void* tag;  // uniquely identifies a request.
  bool ok;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    cq->Next(&tag, &ok);
    GPR_ASSERT(ok);
    static_cast<CallData*>(tag)->Proceed();
  }
}

// Runs DetectService server until shutdown.
void RunServer(const int port, const string& servable_name,
               std::unique_ptr<tensorflow::serving::Manager> manager) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);

  DetectService::AsyncService service;
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<ServerCompletionQueue> cq = builder.AddCompletionQueue();
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";

  DetectServiceImpl service_impl(servable_name, std::move(manager));
  HandleRpcs(&service_impl, &service, cq.get());
}

string get_exe_path() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return string(result, (count > 0) ? count : 0);
}

}  // namespace

int main(int argc, char** argv) {
  // Parse command-line options.
  tensorflow::int32 port = 0;

  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, { tensorflow::Flag("port", &port) });
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }
  if (argc != 2) {
    LOG(FATAL) << "Usage: rcnn_classifier --port=9000 /path/to/exports";
  }
  const string export_base_path(argv[1]);

  // Initialize Caffe subsystem
  tensorflow::serving::CaffeGlobalInit(&argc, &argv);
  tensorflow::serving::CaffeSourceAdapterConfig source_adapter_config;
  {
    auto bundle_cfg = source_adapter_config.mutable_config();
    // enable pycaffe
    bundle_cfg->set_enable_py_caffe(true);
    // add path to pycaffe python module(s)
    bundle_cfg->add_python_path(tensorflow::strings::StrCat(get_exe_path(),
      ".runfiles/tf_serving/tensorflow_serving/servables/caffe/pycaffe"));
    // add path to py-faster-rcnn
    bundle_cfg->add_python_path(export_base_path + "/lib");
    // reshape the image blob
    auto& shape = (*bundle_cfg->mutable_named_initial_shapes())[ "data" ];
    shape.add_dim()->set_size(1);
    shape.add_dim()->set_size(kNumChannels);
    shape.add_dim()->set_size(kImageSizeH);
    shape.add_dim()->set_size(kImageSizeW);
  }
  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleCaffeModelManagerFromBasePath(export_base_path, source_adapter_config, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());

  // Run the service.
  RunServer(port, ready_ids[0].name, std::move(manager));
  return 0;
}
