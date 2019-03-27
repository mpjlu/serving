docker run -p $1:$1 -v /tmp/mnist:/models/mnist -e MODEL_NAME=mnist -t tensorflow/serving:nightly-devel tensorflow_model_server --port=$1 --model_name=mnist --model_base_path=/models/mnist &
