#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

#ifdef WITH_PYTHON_LAYER
#  include "python_prelude.h"
   // embedded python module initialization function
   extern "C" void init_caffe();
#endif

#include "tensorflow_serving/servables/caffe/caffe_py_util.h"

namespace tensorflow {
namespace serving {

bool IsPyCaffeAvailable() {
#ifdef WITH_PYTHON_LAYER
  return true;
#else
  return false;
#endif
}

tensorflow::Status EnsurePyCaffe() {
  if (!IsPyCaffeAvailable()) {
  	return tensorflow::errors::Internal(
  		"Python unavilable in this build configuration");
  }
  else {
  	return Status::OK();
  }
}

tensorflow::Status EnsurePyCaffeSystemPath(const string& path) {
  TF_RETURN_IF_ERROR(EnsurePyCaffe());
  TF_RETURN_IF_ERROR(EnsurePyCaffeInitialized());

#ifdef WITH_PYTHON_LAYER
  auto statement = strings::StrCat(
  	  "import sys\n",
  	  "if not '", path, "' in sys.path: sys.path.append('", path, "')\n"
  	  );

  PyRun_SimpleString(statement.c_str());
#endif

  return Status::OK();
}

tensorflow::Status EnsurePyCaffeInitialized() {
  static bool initialized = false;
  TF_RETURN_IF_ERROR(EnsurePyCaffe());

  if (!initialized) {
#ifdef WITH_PYTHON_LAYER
  	LOG(INFO) << "Initializing Python:\n" << Py_GetVersion();
  	// append the pythohn internal modules with py
  	// the default module search path
  	string path{ Py_GetPath() };
  	// make the caffe module accessible as a builtin
  	if (PyImport_AppendInittab("_caffe", &init_caffe) == -1) {
  	  return tensorflow::errors::Internal("Failed to add PyCaffe builtin module");
  	}
  	// causes a fatal error if initilization fails :(
  	Py_Initialize();
  	// set sys.path to default search path.
  	PySys_SetPath(path.c_str());
  	// append site-specific paths to the module search path
  	// and add other builtins.
    PyRun_SimpleString("import site;site.main()");
    if (PyErr_Occurred() != nullptr) {
    	PyErr_PrintEx(0);
    	return tensorflow::errors::Internal("Python initialization failed.");
    }
    initialized = true;
#endif
  }
  return Status::OK();
}

} // namespace serving
} // namespaces tensorflow