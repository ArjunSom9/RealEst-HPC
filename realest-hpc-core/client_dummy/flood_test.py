import asyncio
import time
import random
import statistics
import grpc

# Note: You must generate these files first using:
# python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/property.proto ../proto/search_request.proto
try:
    import property_pb2
    import property_pb2_grpc
    import search_request_pb2
except ImportError:
    print("Error: Generated proto files not found.")
    print("Run protoc to generate Python gRPC bindings before running this script.")
    exit(1)

# --- Configuration for the Flood Test ---
SERVER_ADDRESS = 'localhost:50051'
VECTOR_DIMENSION = 128  # Size of the embeddings
TOTAL_REQUESTS = 10000  # Start with 10k for local testing, scale to 50k+ for true HPC testing
CONCURRENCY_LIMIT = 500 # Number of concurrent in-flight requests

def generate_random_embedding(dim: int) -> list[float]:
    """Generates a random normalized float array simulating a vector embedding."""
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]

def create_random_ingest_request(property_id: str) -> property_pb2.IngestRequest:
    """Generates a dummy IngestRequest with randomized coordinates and embeddings."""
    return property_pb2.IngestRequest(
        property=property_pb2.Property(
            id=property_id,
            location=property_pb2.GeoLocation(
                latitude=random.uniform(25.0, 49.0),   # Roughly US bounds
                longitude=random.uniform(-125.0, -67.0)
            ),
            embedding=generate_random_embedding(VECTOR_DIMENSION),
            price=random.uniform(100000, 2000000),
            bedrooms=random.randint(1, 6),
            bathrooms=random.randint(1, 4)
        )
    )

def create_random_search_request() -> search_request_pb2.SearchRequest:
    """Generates a dummy SearchRequest to trigger the Phase 2 fused kernels."""
    return search_request_pb2.SearchRequest(
        query_embedding=generate_random_embedding(VECTOR_DIMENSION),
        query_location=property_pb2.GeoLocation(
            latitude=random.uniform(25.0, 49.0),
            longitude=random.uniform(-125.0, -67.0)
        ),
        top_k=10,
        max_radius_meters=5000.0
    )

async def fire_search_request(stub: property_pb2_grpc.RealEstServiceStub, semaphore: asyncio.Semaphore, latencies: list, server_compute_times: list):
    """Fires a single search request and records the round-trip and server-side latency."""
    request = create_random_search_request()
    
    async with semaphore:
        start_time = time.perf_counter()
        try:
            # Execute the async gRPC call
            response = await stub.SearchProperties(request)
            
            # Record metrics
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            server_compute_times.append(response.compute_time_ms)
            
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")

async def run_flood_test():
    print(f"=== RealEst-HPC Distributed Engine Flood Tester ===")
    print(f"Target: {SERVER_ADDRESS}")
    print(f"Requests: {TOTAL_REQUESTS} | Concurrency: {CONCURRENCY_LIMIT}")
    print("---------------------------------------------------")
    
    # Establish async gRPC channel
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = property_pb2_grpc.RealEstServiceStub(channel)
        
        # 1. Warmup: Ingest a few properties
        print("[*] Warming up ingestion service...")
        for i in range(10):
            req = create_random_ingest_request(f"warmup_prop_{i}")
            await stub.IngestProperty(req)
        print("[*] Warmup complete. Beginning search flood test...\n")

        # 2. Setup concurrency control and metric storage
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        latencies = []
        server_compute_times = []
        
        # 3. Create all tasks
        tasks = [
            fire_search_request(stub, semaphore, latencies, server_compute_times)
            for _ in range(TOTAL_REQUESTS)
        ]
        
        # 4. FIRE! Measure total time taken to resolve all futures
        start_test = time.perf_counter()
        await asyncio.gather(*tasks)
        end_test = time.perf_counter()

        # 5. Calculate and display Phase 4 metrics
        total_time_secs = end_test - start_test
        rps = TOTAL_REQUESTS / total_time_secs
        
        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        avg_server_compute = statistics.mean(server_compute_times) if server_compute_times else 0

        print("=== TEST RESULTS ===")
        print(f"Total Time:      {total_time_secs:.2f} seconds")
        print(f"Throughput:      {rps:.2f} Requests / Second")
        print(f"Avg Round-Trip:  {avg_latency:.2f} ms")
        print(f"P99 Round-Trip:  {p99_latency:.2f} ms")
        print(f"Server Compute:  {avg_server_compute:.4f} ms (Targeting sub-millisecond!)")
        print("====================")
        
        if rps > 40000:
            print("🚀 Phase 4 Throughput Goal (50k RPS) is within reach!")
        else:
            print("🔧 Tuning required: Check C++ lock contention or network limits to reach 50k RPS.")

if __name__ == "__main__":
    # Required for Python 3.8+ asyncio on some platforms to handle high socket connections cleanly
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_flood_test())