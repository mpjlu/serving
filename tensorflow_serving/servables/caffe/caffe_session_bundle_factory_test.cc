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

#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"
#include "tensorflow_serving/servables/caffe/example/mnist_sample.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "caffe/blob.hpp"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_config.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class CaffeSessionBundleFactoryTest : public ::testing::Test {
 protected:
  CaffeSessionBundleFactoryTest()
      : export_dir_(test_util::TestSrcDirPath(
            "servables/caffe/example/00000023")) {}

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;

  // Test that a SessionBundle handles a single request for the half plus two
  // model properly. The request has size=2, for batching purposes.
  void TestSingleRequest(CaffeSessionBundle* bundle) {
    const std::vector<float> input = mnist_sample_28x28();
    ASSERT_EQ(28 * 28, input.size());

    caffe::Blob<float>* input_layer = bundle->session->input_blobs()[0];
    caffe::Blob<float>* output_layer = bundle->session->output_blobs()[0];
    {
      ASSERT_EQ(input_layer->count(), input.size() * 64 /* batch */);
      std::copy_n(input.data(), input.size(), input_layer->mutable_cpu_data());
    }

    bundle->session->Forward();
    {
      ASSERT_EQ(output_layer->count(0), 10 * 64 /* batch */);
      /* Copy the output layer to a std::vector */
      const float* begin = output_layer->cpu_data();
      const float* end = begin + output_layer->channels();

      std::vector<float> result(begin, end);

      /* convert result to tensor */
      {
        Tensor expected_output = test::AsTensor<float>(
          { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, { 10 }
        );
        Tensor output = test::AsTensor<float>(
          result, { 10 }
        );

        test::ExpectTensorNear<float>(expected_output, output, 0.05);
      }
    }

    
//    const auto& single_output = outputs.at(0);
//    test::ExpectTensorEqual<float>(expected_output, single_output);
  }
};

TEST_F(CaffeSessionBundleFactoryTest, Basic) {
  const CaffeSessionBundleConfig config;
  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
  std::unique_ptr<CaffeSessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));
  TestSingleRequest(bundle.get());
}

// TEST_F(CaffeSessionBundleFactoryTest, Batching) {
//   SessionBundleConfig config;
//   BatchingParameters* batching_params = config.mutable_batching_parameters();
//   batching_params->mutable_max_batch_size()->set_value(2);
//   batching_params->mutable_max_time_micros()->set_value(
//       10 * 1000 * 1000 /* 10 seconds (should be plenty of time) */);
//   std::unique_ptr<CaffeSessionBundleFactory> factory;
//   TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
//   std::unique_ptr<CaffeSessionBundle> bundle;
//   TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));
//   SessionBundle* bundle_raw = bundle.get();

//   // Run multiple requests concurrently. They should be executed as
//   // 'kNumRequestsToTest/2' batches.
//   {
//     const int kNumRequestsToTest = 10;
//     std::vector<std::unique_ptr<Thread>> request_threads;
//     for (int i = 0; i < kNumRequestsToTest; ++i) {
//       request_threads.push_back(
//           std::unique_ptr<Thread>(Env::Default()->StartThread(
//               ThreadOptions(), strings::StrCat("thread_", i),
//               [this, bundle_raw] { this->TestSingleRequest(bundle_raw); })));
//     }
//   }
// }

// TEST_F(CaffeSessionBundleFactoryTest, BatchingConfigError) {
//   CaffeSessionBundleConfig config;
//   BatchingParameters* batching_params = config.mutable_batching_parameters();
//   batching_params->mutable_max_batch_size()->set_value(2);
//   // The last entry in 'allowed_batch_sizes' is supposed to equal
//   // 'max_batch_size'. Let's violate that constraint and ensure we get an error.
//   batching_params->add_allowed_batch_sizes(1);
//   batching_params->add_allowed_batch_sizes(3);
//   std::unique_ptr<SessionBundleFactory> factory;
//   EXPECT_FALSE(SessionBundleFactory::Create(config, &factory).ok());
// }

// TEST_F(CaffeSessionBundleFactoryTest, EstimateResourceRequirementWithBadExport) {
//   const CaffeSessionBundleConfig config;
//   std::unique_ptr<CaffeSessionBundleFactory> factory;
//   TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));
//   ResourceAllocation resource_requirement;
//   const Status status = factory->EstimateResourceRequirement(
//       "/a/bogus/export/dir", &resource_requirement);
//   EXPECT_FALSE(status.ok());
// }

// TEST_F(CaffeSessionBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
//   const uint64 kVariableFileSize = 169;
//   const uint64 expected_ram_requirement =
//       kVariableFileSize * CaffeSessionBundleFactory::kResourceEstimateRAMMultiplier +
//       CaffeSessionBundleFactory::kResourceEstimateRAMPadBytes;
//   ResourceAllocation want;
//   ResourceAllocation::Entry* ram_entry = want.add_resource_quantities();
//   Resource* ram_resource = ram_entry->mutable_resource();
//   ram_resource->set_device(device_types::kMain);
//   ram_resource->set_kind(resource_kinds::kRamBytes);
//   ram_entry->set_quantity(expected_ram_requirement);

//   const SessionBundleConfig config;
//   std::unique_ptr<SessionBundleFactory> factory;
//   TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));
//   ResourceAllocation got;
//   TF_ASSERT_OK(factory->EstimateResourceRequirement(export_dir_, &got));

//   EXPECT_THAT(got, EqualsProto(want));
// }

}  // namespace
}  // namespace serving
}  // namespace tensorflow
