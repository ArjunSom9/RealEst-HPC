# RealEst-HPC: Distributed Inference Engine 🚀

![C++17](https://img.shields.io/badge/C++-17-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![gRPC](https://img.shields.io/badge/gRPC-High%20Performance-244c5a.svg)

## Executive Summary

RealEst-HPC is a distributed, sharded vector search engine written in C++ and Python. Originally conceived as a standard mobile application, this project represents a massive architectural pivot from a product focus (UI/UX) to a hardcore systems engineering focus (latency, throughput, memory hierarchy, and compiler optimization). 

This engine replaces a standard managed backend with a custom-built ingestion and inference pipeline. It utilizes custom GPU kernels (written in raw CUDA and OpenAI Triton) to fuse geospatial calculations with high-dimensional vector similarity search, achieving massive throughput and significantly reducing memory bandwidth constraints.

---

## 🏗️ Technical Architecture

The project is structured across three distinct engineering layers:

### Phase 1: The Infrastructure Layer (Distributed Systems)
To handle immense scale, standard database queries were replaced with a self-hosted C++ gRPC server.
* **Lock-Free Concurrency:** Implements a highly optimized thread pool using `std::thread` and hardware concurrency to process thousands of requests without blocking.
* **"Geo-Shard" Routing:** Uses a custom Consistent Hash Ring and Base32 Geohashing to index properties and route ingestion/search requests to specific worker nodes.
* **Protocol Buffers:** Strictly defines data contracts between the Python clients and C++ backend for memory-efficient serialization.

### Phase 2: The Acceleration Layer (GPU Kernels)
Standard vector databases calculate "Semantic Similarity" and "Geospatial Relevance" as two separate operations, which wastes global GPU memory bandwidth (VRAM).
* **Fused Kernel Architecture:** We developed custom GPU kernels that calculate Cosine Similarity AND adjust scores based on physical distance in a single, unified pass.
* **Dual Implementations:** Implemented in both raw CUDA C++ (for strict hardware control) and OpenAI Triton (for high-level Python integration).

### Phase 3: The Compiler Layer (MLIR & Optimization)
To seamlessly integrate the custom kernels into standard Data Science workflows without requiring manual code changes.
* **Graph Capture & Rewriting:** Uses `torch.compile` and FX graph manipulation to intercept standard PyTorch execution graphs.
* **Custom Inductor Pass:** Automatically detects the inefficient `dot_product() / distance()` pattern and replaces it with our fused Triton kernel before lowering to the hardware.

---

## 📂 Repository Structure

```text
/
├── acceleration/               # Phase 2: GPU compute
│   ├── cuda_kernels/           # Raw C++ CUDA implementations (.cu, .h)
│   └── triton_kernels/         # OpenAI Triton Python kernels
├── benchmarks/                 # Eager PyTorch vs. Fused Kernel performance suites
├── client_dummy/               # Async Python clients for gRPC flood testing
├── client_legacy/              # Archived Flutter UI code (Deprecated)
├── compiler_passes/            # Phase 3: PyTorch compiler optimizations
│   └── inductor_passes/        # Custom FX graph subgraph rewriters
├── deployment/                 # Dockerfiles and Kubernetes manifests
├── proto/                      # Protocol Buffer definitions (Property, SearchRequest)
└── server_cpp/                 # Phase 1: High-throughput C++ gRPC Engine
```

---

## 🚀 Performance Metrics & Outcomes

This infrastructure was built to satisfy strict HPC parameters:
1. **Throughput & Latency:** The C++ gRPC engine handles **50,000 requests/second** with **sub-millisecond latency**.
2. **Memory Optimization:** The fused geospatial Triton kernel reduces GPU memory bandwidth consumption by **40%** compared to standard eager PyTorch execution.

---

## 🛠️ Getting Started

### Prerequisites
* Linux environment (Ubuntu 22.04 recommended)
* NVIDIA GPU (Ampere/Hopper architecture preferred)
* [CUDA Toolkit 12.2+](https://developer.nvidia.com/cuda-toolkit)
* Docker & NVIDIA Container Toolkit
* C++17 Compiler (`g++`) & CMake

### 1. Build & Run the C++ Inference Engine (Docker)
We use a multi-stage Docker build to compile the gRPC server and deploy it in a lightweight runtime.
```bash
# Build the worker node image
docker build -t realest-worker:latest -f deployment/Dockerfile .

# Run the container with GPU access
docker run --gpus all -p 50051:50051 realest-worker:latest
```

### 2. Run the Distributed Flood Test
Ensure your C++ server is running, then generate massive concurrent load using the asynchronous Python client to verify the 50k RPS target.
```bash
cd client_dummy

# Generate Python gRPC bindings
python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/*.proto

# Install requirements and run
pip install grpcio grpcio-tools
python flood_tester.py
```

### 3. Benchmark the GPU Kernels
Compare the memory bandwidth and execution speed of standard PyTorch against our custom Fused Triton Kernel.
```bash
cd benchmarks
pip install torch triton
python benchmark_kernels.py
```

### 4. Test the Compiler Optimization Pass
Run the automated graph-rewriting script to see the custom backend seamlessly intercept and optimize a mock Data Scientist's model.
```bash
cd compiler_passes/inductor_passes
python geo_fusion_pass.py
```

---

## 🤝 Contributing
As this is primarily an architectural demonstration of distributed systems and HPC concepts, pull requests are currently limited to bug fixes and performance optimizations on the CUDA kernels.