#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <thread>

// gRPC headers
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

// Generated from your proto files
// #include "property.pb.h"
// #include "search_request.pb.h"
// #include "property.grpc.pb.h"

// In a real build system (CMake/Bazel), these would be proper .h inclusions.
// For the sake of this architectural demonstration, we include the implementations.
#include "worker_pool.cc"
#include "shard_manager.cc"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

// Namespaces from the proto definition
using realest::hpc::RealEstService;
using realest::hpc::IngestRequest;
using realest::hpc::IngestResponse;
using realest::hpc::SearchRequest;
using realest::hpc::SearchResponse;
using realest::hpc::SearchResult;

namespace realest {
namespace hpc {

/**
 * @class RealEstServiceImpl
 * @brief The actual implementation of the gRPC service defined in property.proto.
 * Ties together the ShardManager (routing) and WorkerPool (concurrency) to 
 * achieve the Phase 1 50k req/sec throughput target.
 */
class RealEstServiceImpl final : public RealEstService::Service {
public:
    RealEstServiceImpl() 
        // Initialize the thread pool with the number of hardware threads available
        : worker_pool_(std::thread::hardware_concurrency()) 
    {
        // For demonstration, we simulate registering local docker nodes on startup.
        // In a real K8s deployment, nodes would register themselves via a discovery service.
        std::cout << "[Server] Initializing Shard Manager...\n";
        shard_manager_.registerWorker("node-0.realest.internal:50051");
        shard_manager_.registerWorker("node-1.realest.internal:50051");
        shard_manager_.registerWorker("node-2.realest.internal:50051");
    }

    /**
     * @brief RPC implementation for IngestProperty.
     * Offloads the Geohashing and network forwarding to the thread pool to keep 
     * the main gRPC IO threads unblocked.
     */
    Status IngestProperty(ServerContext* context, const IngestRequest* request, 
                          IngestResponse* reply) override {
        
        // Push the ingestion task to our custom thread pool
        auto ingest_task = [this, request]() -> std::string {
            const auto& prop = request->property();
            
            // 1. Calculate the Geohash and find the target worker node
            std::string target_node = shard_manager_.routeIngestion(prop);
            
            // 2. In a fully distributed setup, you would now use a gRPC client 
            // to forward this property to `target_node`. 
            // For now, we simulate saving it to local memory if this *is* the target node.
            
            return "Successfully routed property " + prop.id() + " to " + target_node;
        };

        // Enqueue the task and wait for completion. 
        // (Note: To achieve absolute maximum throughput, you would use gRPC's Async API
        // instead of the sync API shown here, but this demonstrates the pool integration).
        try {
            std::future<std::string> result_future = worker_pool_.enqueue(ingest_task);
            std::string msg = result_future.get();
            
            reply->set_success(true);
            reply->set_message(msg);
            return Status::OK;
        } catch (const std::exception& e) {
            reply->set_success(false);
            reply->set_message(e.what());
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    /**
     * @brief RPC implementation for SearchProperties.
     * This is where the Phase 2 GPU acceleration will be triggered.
     */
    Status SearchProperties(ServerContext* context, const SearchRequest* request, 
                            SearchResponse* reply) override {
        
        // Start high-resolution timer for Phase 4 metrics tracking
        auto start_time = std::chrono::high_resolution_clock::now();

        auto search_task = [this, request, reply]() {
            double lat = request->query_location().latitude();
            double lon = request->query_location().longitude();
            
            // 1. Determine which nodes hold the relevant data
            std::vector<std::string> target_nodes = shard_manager_.routeSearch(
                lat, lon, request->max_radius_meters()
            );

            // 2. PHASE 2 INTEGRATION POINT: 
            // This is where you will invoke your fused Triton/CUDA kernel.
            // e.g., run_fused_geo_score_kernel(request->query_embedding().data(), ...);
            
            // Simulate the GPU compute time for demonstration purposes (e.g., 0.5ms)
            std::this_thread::sleep_for(std::chrono::microseconds(500));

            // Mocking a result returned by the GPU kernel
            SearchResult* result = reply->add_results();
            result->set_property_id("mock_property_883A");
            result->set_fused_score(0.94f); // High combined geo+semantic score
        };

        // Execute the heavy search task in the worker pool
        worker_pool_.enqueue(search_task).get();

        // Calculate and attach the compute time metric
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end_time - start_time;
        
        reply->set_compute_time_ms(duration.count());

        return Status::OK;
    }

private:
    WorkerPool worker_pool_;
    ShardManager shard_manager_;
};

} // namespace hpc
} // namespace realest

// ============================================================================
// SERVER BOOTSTRAP
// ============================================================================

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    realest::hpc::RealEstServiceImpl service;

    ServerBuilder builder;
    
    // Listen on the given address without any authentication mechanism for now
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    
    // Register "service" as the instance through which we'll communicate with clients
    builder.RegisterService(&service);
    
    // Finally assemble the server
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "[Server] RealEst-HPC Inference Engine listening on " << server_address << std::endl;
    std::cout << "[Server] Operating with " << std::thread::hardware_concurrency() 
              << " hardware threads in the worker pool.\n";

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv) {
    // Enable gRPC reflection so tools like grpcurl or Postman can inspect the API
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    
    RunServer();
    return 0;
}