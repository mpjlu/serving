/* Copyright IBM Corp. All Rights Reserved. */
#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session.h"

#include <vector>
#include <array>

namespace rcnn
{

struct Detection {
  std::array<int, 4> roi_rect;
  int class_idx;
  float score;

  inline std::string DebugString() const {
    return tensorflow::strings::StrCat(
      "Detection { class: ", class_idx, ", score: ", score, ", rect: [",
        roi_rect[0], " ", roi_rect[1], " ", roi_rect[2], " ", roi_rect[3], "] }");
  }
};

tensorflow::Status BatchBGRMeansSubtract(
    tensorflow::Tensor* im_batch_blob);

tensorflow::Status RunClassification(
    const tensorflow::Tensor& im_blob,
    const tensorflow::Tensor& im_info,
    tensorflow::Session* session,
    tensorflow::Tensor* pred_boxes,
    tensorflow::Tensor* scores,
    tensorflow::Tensor* class_labels);

tensorflow::Status ProcessDetections(
    const tensorflow::Tensor* pred_boxes,
    const tensorflow::Tensor* scores,
    std::vector<Detection>* dets);

} // namespace rcnn
