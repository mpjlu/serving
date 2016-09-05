/* Copyright IBM Corp. All Rights Reserved. */
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/tensor_testutil.h"

#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_config.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_py_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

inline string TestPyDirPath() {
  return test_util::TestSrcDirPath("servables/caffe/pycaffe");
}

class CaffeSessionBundleFactoryPyTest : public ::testing::Test {
 protected:
  CaffeSessionBundleFactoryPyTest()
      : export_dir_(test_util::TestSrcDirPath(
          "servables/caffe/test_data/py_layers/00000001"))
      , input_sample_(0) {
    input_sample_.resize(9 * 8, 7.0);
  }

  const string export_dir_;
  std::vector<float> input_sample_;

  // Test that a SessionBundle handles a single request for the 10**3
  // model correctly.
  void TestSingleRequest(CaffeSessionBundle* bundle) {
    ASSERT_EQ(9 * 8, input_sample_.size());
    gtl::ArraySlice<float> input_slice(input_sample_);

    std::vector<std::pair<string, Tensor>> inputs {
      { "data", test::AsTensor(input_slice, {1, 9, 8}) }
    };
    std::vector<Tensor> outputs;

    TF_ASSERT_OK(bundle->session->Run(inputs, {"three"}, {}, &outputs));
    ASSERT_EQ(outputs.size(), 1);

    auto vec = outputs[0].flat<float>();
    ASSERT_EQ(vec.size(), input_sample_.size());

    for (int i = 0; i < input_sample_.size(); ++i) {
      ASSERT_EQ(vec(i), input_sample_[i] * pow(10, 3));
    }
  }
};

TEST_F(CaffeSessionBundleFactoryPyTest, PythonLayer) {
#ifndef WITH_PYTHON_LAYER
  // do nothing if pycaffe is not enabled
  ASSERT_FALSE(IsPyCaffeAvailable());
  LOG(INFO) << "[CaffeSessionBundleFactoryPyTest] skipped because pycaffe unavailable";

#else
  ASSERT_TRUE(IsPyCaffeAvailable());

  CaffeSessionBundleConfig config;
  config.set_enable_py_caffe(true);
  config.add_python_path(TestPyDirPath());
  config.add_python_path(export_dir_);

  std::unique_ptr<CaffeSessionBundleFactory> factory;
  TF_ASSERT_OK(CaffeSessionBundleFactory::Create(config, &factory));

  std::unique_ptr<CaffeSessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));

  TestSingleRequest(bundle.get());
#endif
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow