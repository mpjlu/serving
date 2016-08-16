#pragma once

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// Returns true if Caffe was built with Python layer.
bool IsPyCaffeAvailable();

// Ensure python is loaded and initialized with the caffe
// wrapper module.
tensorflow::Status EnsurePyCaffeInitialized();

// Ensure the given path is included in the python
// module search path (sys.path)
tensorflow::Status EnsurePyCaffeSystemPath(const string& path);

} // namespace serving
} // namespaces tensorflow