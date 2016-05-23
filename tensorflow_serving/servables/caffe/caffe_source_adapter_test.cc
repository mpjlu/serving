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

#include <memory>
#include <unordered_map>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "caffe/net.hpp"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/test_util/source_adapter_test_util.h"
#include "tensorflow_serving/servables/caffe/caffe_source_adapter.pb.h"
#include "tensorflow_serving/util/any_ptr.h"

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace tensorflow {
namespace serving {
namespace {

using Net = caffe::Net<float>;

// Writes the given hashmap to a file.
Status WriteCaffeToFile(const CaffeSourceAdapterConfig::Format format,
                        const string& file_name, const std::string& model_definition) {
  WritableFile* file_raw;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(file_name, &file_raw));
  std::unique_ptr<WritableFile> file(file_raw);
  file->Append(model_definition);
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

TEST(CaffeSourceAdapter, Basic) {
  const auto format = CaffeSourceAdapterConfig::SIMPLE_CSV;
  const string file = io::JoinPath(testing::TmpDir(), "Basic");

  TF_ASSERT_OK(
      WriteCaffeToFile(format, file, R"(
          name: "LeNet"
          layer {
            name: "data"
            type: "Input"
            top: "data"
            input_param { shape: { dim: 64 dim : 1 dim : 28 dim : 28 } }
          }
      )"));

  CaffeSourceAdapterConfig config;
  config.set_format(format);

  auto adapter =
      std::unique_ptr<CaffeSourceAdapter>(new CaffeSourceAdapter(config));
  
  ServableData<std::unique_ptr<Loader>> loader_data =
      test_util::RunSourceAdapter(file, adapter.get());
  
  TF_ASSERT_OK(loader_data.status());

  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();

  TF_ASSERT_OK(loader->Load(ResourceAllocation()));

  const Net* model = loader->servable().get<Net>();
  EXPECT_EQ(model->name(), "LeNet");
  
  loader->Unload();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
