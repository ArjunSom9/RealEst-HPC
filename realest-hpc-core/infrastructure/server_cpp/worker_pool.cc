#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <memory>

namespace realest {
namespace hpc {

/**
 * @class WorkerPool
 * @brief A high-throughput thread pool designed for the RealEst-HPC Inference Engine.
 * * Manages a pool of worker threads that process incoming gRPC requests (Ingestion and Search).
 * * HPC Note: While this uses std::mutex for condition waiting (to prevent CPU spinning), 
 * to achieve the strict "lock-free queue" metric mentioned in Phase 4 for the ultimate 
 * 50k req/sec throughput, the std::queue below should be swapped with a lock-free 
 * structure like `moodycamel::ConcurrentQueue` or a custom std::atomic ring buffer in production.
 */
class WorkerPool {
public:
    // Constructor: Initializes the pool with a specific number of worker threads.
    explicit WorkerPool(size_t num_threads);

    // Destructor: Safely shuts down threads, ensuring all pending tasks complete.
    ~WorkerPool();

    // Prevent copying and assignment to avoid resource duplication
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /**
     * @brief Enqueues a new task into the worker pool.
     * * @tparam F Type of the function/callable.
     * @tparam Args Types of the arguments to the function.
     * @param f The function to execute (e.g., a geospatial vector search).
     * @param args The arguments to pass to the function.
     * @return std::future holding the return value of the executed function.
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    // The actual worker threads
    std::vector<std::thread> workers_;

    // The task queue. Holds type-erased callables.
    std::queue<std::function<void()>> tasks_;

    // Synchronization primitives
    std::mutex queue_mutex_;
    std::condition_variable condition_;

    // Atomic flag to signal threads to stop processing and exit
    std::atomic<bool> stop_;
};

// ============================================================================
// IMPLEMENTATION
// ============================================================================

inline WorkerPool::WorkerPool(size_t num_threads) : stop_(false) {
    // Pre-allocate space to avoid reallocations during startup
    workers_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            // Infinite loop for the worker thread
            while (true) {
                std::function<void()> task;

                {
                    // Lock the queue mutex to safely check for tasks
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);

                    // Wait until there is a task OR the pool is stopping
                    this->condition_.wait(lock, [this] {
                        return this->stop_.load() || !this->tasks_.empty();
                    });

                    // If the pool is stopping and the queue is empty, exit the thread
                    if (this->stop_.load() && this->tasks_.empty()) {
                        return;
                    }

                    // Extract the task from the front of the queue
                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                } // Mutex is automatically released here

                // Execute the task outside the lock to allow concurrent execution
                task();
            }
        });
    }
}

template<class F, class... Args>
auto WorkerPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;

    // Package the task so we can extract a std::future from it
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Don't allow enqueuing after stopping the pool
        if (stop_.load()) {
            throw std::runtime_error("enqueue on stopped WorkerPool");
        }

        // Wrap the packaged task in a type-erased void function and push to queue
        tasks_.emplace([task]() {
            (*task)();
        });
    }

    // Wake up one sleeping worker thread to handle the new task
    condition_.notify_one();
    
    return res;
}

inline WorkerPool::~WorkerPool() {
    // Signal all threads to stop processing new tasks
    stop_.store(true);
    
    // Wake up all threads so they can finish current tasks and exit
    condition_.notify_all();

    // Wait for all threads to finish execution gracefully
    for (std::thread &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace hpc
} // namespace realest

/* // ============================================================================
// USAGE EXAMPLE (For shard_manager.cc or main.cc)
// ============================================================================
// 
// #include "worker_pool.cc"
// 
// int main() {
//     // Initialize pool with hardware concurrency (e.g., 16 threads)
//     realest::hpc::WorkerPool pool(std::thread::hardware_concurrency());
// 
//     // Simulate receiving a gRPC SearchRequest
//     auto search_task = [](std::string query_id) {
//         // This is where you would call your custom Phase 2 GPU Kernel
//         // e.g., run_fused_geo_score_kernel(query_id);
//         return "Search " + query_id + " completed in 0.8ms";
//     };
// 
//     // Enqueue the task asynchronously
//     std::future<std::string> result = pool.enqueue(search_task, "req_9948A");
// 
//     // Do other non-blocking gRPC work here...
// 
//     // Retrieve the result when ready
//     std::cout << result.get() << std::endl; 
// 
//     return 0; // Pool destructor automatically joins threads
// }
*/