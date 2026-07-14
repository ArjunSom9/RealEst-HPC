#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "shard_manager.cc"
// Assuming generated headers are available in the build environment
// #include "property.grpc.pb.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

using realest::hpc::RealEstService;
using realest::hpc::SearchRequest;
using realest::hpc::SearchResponse;

namespace realest {
namespace hpc {

/**
 * @class CallData
 * @brief Explicitly manages the memory and lifecycle of a single RPC.
 * This satisfies Section 3: "explicitly manages the lifecycle and memory state 
 * of every Remote Procedure Call through a custom CallData object."
 */
class CallData {
public:
    CallData(RealEstService::AsyncService* service, ServerCompletionQueue* cq, ShardManager* shard_manager)
        : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), shard_manager_(shard_manager) {
        Proceed();
    }

    void Proceed() {
        if (status_ == CREATE) {
            // Transition to PROCESS state.
            status_ = PROCESS;
            
            // Request the gRPC runtime to start listening for a SearchProperties RPC.
            // We pass `this` as the unique tag.
            service_->RequestSearchProperties(&ctx_, &request_, &responder_, cq_, cq_, this);
        } else if (status_ == PROCESS) {
            // Spawn a NEW CallData instance to serve the next client immediately,
            // preventing the server from blocking.
            new CallData(service_, cq_, shard_manager_);

            // --- ACTUAL BUSINESS LOGIC ---
            // 1. Spatial Routing (Section 4)
            double lat = request_.query_location().latitude();
            double lon = request_.query_location().longitude();
            
            std::vector<std::string> target_nodes = shard_manager_->routeSearch(
                lat, lon, request_.max_radius_meters()
            );

            // 2. Vector Engine Retrieval (Section 5)
            // Here is where the FlatHNSWIndex (flat_hnsw.cc) would be queried.
            
            // Constructing mock response
            reply_.set_compute_time_ms(0.8f); 
            
            // Transition to FINISH and send the response back to the client.
            status_ = FINISH;
            responder_.Finish(reply_, Status::OK, this);
        } else {
            // status_ == FINISH
            // The RPC is fully complete. We manage our own memory and delete ourselves.
            GPR_ASSERT(status_ == FINISH);
            delete this;
        }
    }

private:
    RealEstService::AsyncService* service_;
    ServerCompletionQueue* cq_;
    ServerContext ctx_;
    
    SearchRequest request_;
    SearchResponse reply_;
    ServerAsyncResponseWriter<SearchResponse> responder_;
    ShardManager* shard_manager_;

    // The state machine defining the lifecycle of the RPC
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;
};

class AsyncServerImpl {
public:
    ~AsyncServerImpl() {
        server_->Shutdown();
        cq_->Shutdown();
    }

    void Run() {
        std::string server_address("0.0.0.0:50051");
        ServerBuilder builder;
        
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);
        
        // This replaces the thread pool. The CompletionQueue is the heart of the async model.
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        std::cout << "[Async Server] Listening on " << server_address << std::endl;

        // Register dummy workers for the shard manager
        shard_manager_.registerWorker("node-0.internal");
        shard_manager_.registerWorker("node-1.internal");

        HandleRpcs();
    }

private:
    void HandleRpcs() {
        // Spawn the first CallData instance to start listening
        new CallData(&service_, cq_.get(), &shard_manager_);
        
        void* tag;  // Uniquely identifies a request.
        bool ok;
        
        // Block waiting to read the next event from the completion queue.
        // In a true multi-core setup (Section 3), you would run this loop 
        // on exactly one pinned thread per physical CPU core.
        while (true) {
            GPR_ASSERT(cq_->Next(&tag, &ok));
            
            // Ensure the event is valid (e.g., client didn't disconnect unexpectedly)
            if (!ok) {
                // In production, handle graceful cleanup here
                continue; 
            }
            
            // Cast the tag back to our CallData state machine and advance it
            static_cast<CallData*>(tag)->Proceed();
        }
    }

    std::unique_ptr<ServerCompletionQueue> cq_;
    RealEstService::AsyncService service_;
    std::unique_ptr<Server> server_;
    ShardManager shard_manager_;
};

} // namespace hpc
} // namespace realest

int main(int argc, char** argv) {
    realest::hpc::AsyncServerImpl server;
    server.Run();
    return 0;
}