# Use the specified base image
FROM openfabric/tee-python-cpu:latest

# Create and set the working directory
RUN mkdir /application
WORKDIR /application

# Copy the current directory contents into the container
COPY . .

# Install poetry and dependencies without dev dependencies
RUN poetry install -vvv --without dev

RUN apt update && apt install -y git apt-utils

# Expose the necessary ports
EXPOSE 5500

RUN pip uninstall -y openfabric-pysdk

# Install necessary packages and libraries
RUN apt-get update && \
    apt-get install -y libopenblas-dev git wget make

# Clone the llama.cpp repository and build it
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make LLAMA_OPENBLAS=1 -j$(nproc)

# Download the model file into the models directory
RUN mkdir -p ./llama.cpp/models && \
    wget -nv https://grigoryevko-openfabric.s3.eu-north-1.amazonaws.com/ggml-model-q4_0.gguf -O ./llama.cpp/models/ggml-model-q4_0.gguf

# Install Python packages
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN pip install llama-cpp-python git+https://github.com/UCT10/openfabric_pysdk.git

# Set the command to start the container
CMD ["bash", "start.sh"]
