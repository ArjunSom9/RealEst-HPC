# ==============================================================================
# STAGE 1: BUILD ENVIRONMENT
# Uses the CUDA development image to get access to nvcc, g++, and build tools.
# ==============================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

# Prevent tzdata and other interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install C++ build tools, Protobuf, and gRPC dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    protobuf-compiler \
    libprotobuf-dev \
    protobuf-compiler-grpc \
    libgrpc++-dev \
    libgrpc++-reflection-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy the Protocol Buffers and the C++ Source Code
COPY proto/ /app/proto/
COPY server_cpp/ /app/server_cpp/

# Create a build directory
WORKDIR /app/build

# 1. Generate the C++ gRPC and Protobuf bindings
RUN protoc -I=/app/proto \
    --cpp_out=. \
    --grpc_out=. \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
    /app/proto/property.proto /app/proto/search_request.proto

# 2. Compile the high-throughput worker node
# -O3: Maximum optimization for HPC
# -std=c++17: Required for std::shared_mutex in your shard_manager.cc
RUN g++ -O3 -std=c++17 \
    -I. -I/app/server_cpp \
    /app/server_cpp/main.cc property.pb.cc property.grpc.pb.cc search_request.pb.cc search_request.grpc.pb.cc \
    -o realest_worker \
    -lgrpc++_reflection -lgrpc++ -lprotobuf -lpthread -ldl

# ==============================================================================
# STAGE 2: RUNTIME ENVIRONMENT
# Uses the lighter CUDA runtime image to keep the image size minimal.
# ==============================================================================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only the runtime libraries required for gRPC and Protobuf
RUN apt-get update && apt-get install -y \
    libprotobuf32 \
    libgrpc++1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/build/realest_worker /app/realest_worker

# Expose the gRPC port defined in main.cc
EXPOSE 50051

# Run the inference engine worker
CMD ["./realest_worker"]