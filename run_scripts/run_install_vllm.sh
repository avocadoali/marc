start_time=$(date +%s)
nproc=16

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export MAKEFLAGS="-j$(nproc)"

pip install  vllm@git+https://github.com/ekinakyurek/vllm.git@ekin/torchtunecompat



end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
