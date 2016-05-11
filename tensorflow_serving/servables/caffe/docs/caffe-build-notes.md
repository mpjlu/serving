# Caffe build/installation notes (ubuntu 15)

## OpenBLAS 

    > sudo apt-get install libopenblas-dev

    installs OpenBLAS => /usr/lib/openblas-base
                      => /usr/include/openblas

### notes: The OpenBLAS version is contained in `cat /usr/include/openblas/openblas_config.h`:

    #define OPENBLAS_VERSION " OpenBLAS 0.2.14 "



## Protobuf 3.0.0-b2

    > wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.tar.gz && 
      # untar 
      tar zxvf protobuf-cpp-3.0.0-beta-2.tar.gz && \
      # delete tar
      rm protobuf-cpp-3.0.0-beta-2.tar.gz && \
      # build
      cd protobuf-3.0.0-beta-2 && \
      ./configure && \
      make && \
      make check && \
      make install && \
      make clean

    installs protobuf in: /usr/local/lib


## Boost

    > sudo apt-get install --no-install-recommends libboost-all-dev


## Other deps

    > sudo apt-get install libleveldb-dev libsnappy-dev libhdf5-serial-dev
    > sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev


## Build

### configure

    > cmake .. -DCAFFE_TARGET_SOVERSION:string="1" \
        -DCPU_ONLY=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBLAS:string="open" \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_python=OFF \
        -DBUILD_python_layer=OFF \
        -DUSE_OPENCV=OFF

    -- ******************* Caffe Configuration Summary *******************
    -- General:
    --   Version           :   1.0.0-rc3
    --   Git               :   rc3-142-gaf04325
    --   System            :   Linux
    --   C++ compiler      :   /usr/bin/c++
    --   Release CXX flags :   -O3 -DNDEBUG -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
    --   Debug CXX flags   :   -g -fPIC -Wall -Wno-sign-compare -Wno-uninitialized
    --   Build type        :   Release
    --
    --   BUILD_SHARED_LIBS :   ON
    --   BUILD_python      :   ON
    --   BUILD_matlab      :   OFF
    --   BUILD_docs        :   ON
    --   CPU_ONLY          :   ON
    --   USE_OPENCV        :   OFF
    --   USE_LEVELDB       :   ON
    --   USE_LMDB          :   ON
    --   ALLOW_LMDB_NOLOCK :   OFF
    --
    -- Dependencies:
    --   BLAS              :   Yes (open)
    --   Boost             :   Yes (ver. 1.58)
    --   glog              :   Yes
    --   gflags            :   Yes
    --   protobuf          :   Yes (ver. 3.0.0)
    --   lmdb              :   Yes (ver. 0.9.15)
    --   LevelDB           :   Yes (ver. 1.18)
    --   Snappy            :   Yes (ver. 1.1.3)
    --   CUDA              :   No
    --
    -- Python:
    --   Interpreter       :   /usr/bin/python2.7 (ver. 2.7.10)
    --   Libraries         :   /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.10)
    --   NumPy             :   /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.8.2)
    --
    -- Documentaion:
    --   Doxygen           :   No
    --   config_file       :
    --
    -- Install:
    --   Install path      :   /media/sf_data-share/caffe/build_ubuntu/install
    --
    -- Configuring done

### Build

    cmake --build .


## Install (cmake)

    > cmake --build . --target install


## link/compilation flags for other projects (not cmake based):

    > cmake --find-package -DMODE=[LINK/COMPILE] -DLANGUAGE=CXX -DCOMPILER_ID=GNU -DNAME=Caffe