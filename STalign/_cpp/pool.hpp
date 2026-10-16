// Small persistent std::thread pool with a blocking parallel_for.
//
// * No OpenMP: avoids the "OMP: Error #15" clash with the libomp that torch
//   ships on macOS, and builds unchanged with clang, gcc and MSVC.
// * Completion is tracked per *task*, not per worker: the caller (which also
//   executes tasks) only waits until every task has finished.  A worker that is
//   descheduled by the OS (machine oversubscribed, efficiency cores) never
//   delays a parallel region it did not take part in.
// * Tasks are claimed with a CAS on a 64-bit word (job id << 32 | next task),
//   so a late worker can never claim a task of a job it did not read the
//   descriptor of (descriptors are double-buffered by job-id parity).
// * Workers spin briefly (300 us) after each job so back-to-back parallel
//   regions inside one LDDMM iteration do not pay a condition-variable wake-up;
//   after that they sleep (no busy CPU while Python is idle).
// * An exception thrown by a task is re-thrown in the calling thread once all
//   tasks of the region have finished.
// * Results never depend on which thread ran a task: all reductions in the
//   engine are per task and are combined in task order afterwards.
// * Fork safety: global_pool() hands out one pool per process.  A child created
//   by fork() has none of the parent's worker threads, so the inherited pool is
//   abandoned there and a fresh one is created on first use.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace stalign {

inline void cpu_relax() {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
  asm volatile("yield" ::: "memory");
#elif defined(_M_ARM64)
  __yield();
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

inline long process_id() {
#if defined(_WIN32)
  return (long)_getpid();
#else
  return (long)getpid();
#endif
}

class ThreadPool {
 public:
  explicit ThreadPool(int n = 1) : pid_(process_id()) { resize(n); }
  ~ThreadPool() { stop_workers(); }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return nthreads_; }
  long owner_pid() const { return pid_; }

  // Total number of threads, including the calling thread.  Waits for a running
  // parallel region to finish first.
  void resize(int n) {
    std::lock_guard<std::mutex> serial(call_m_);
    if (n < 1) n = 1;
    if (n == nthreads_ && (int)workers_.size() == n - 1) return;
    stop_workers();
    nthreads_ = n;
    stop_.store(false);
    for (int i = 0; i < n - 1; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  // Calls f(i) for i in [0, n) and blocks until all calls returned.  Calls from
  // different threads (e.g. two Python threads running LDDMM at once) are
  // serialised; f must not call parallel_for itself.
  template <class F>
  void parallel_for(int64_t n, F&& f) {
    if (n <= 0) return;
    std::lock_guard<std::mutex> serial(call_m_);
    if (nthreads_ <= 1 || n == 1) {
      for (int64_t i = 0; i < n; ++i) f(i);
      return;
    }
    using Fn = typename std::remove_reference<F>::type;
    const uint64_t id = (job_.load(std::memory_order_relaxed) >> 32) + 1;
    Desc& d = desc_[id & 1];
    d.ctx.store((void*)&f, std::memory_order_relaxed);
    d.fn.store(&call<Fn>, std::memory_order_relaxed);
    d.ntasks.store(n, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    job_.store(id << 32, std::memory_order_seq_cst);  // publish
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
      std::lock_guard<std::mutex> lk(m_);
      cv_.notify_all();
    }
    const int64_t mine = run_tasks(id);
    if (mine) done_.fetch_add(mine, std::memory_order_acq_rel);
    while (done_.load(std::memory_order_acquire) != n) cpu_relax();
    if (error_) {
      std::exception_ptr e = error_;
      error_ = nullptr;
      std::rethrow_exception(e);
    }
  }

 private:
  struct Desc {
    std::atomic<void*> ctx{nullptr};
    std::atomic<void (*)(void*, int64_t)> fn{nullptr};
    std::atomic<int64_t> ntasks{0};
  };

  template <class Fn>
  static void call(void* c, int64_t i) {
    (*(Fn*)c)(i);
  }

  // claim and run tasks of job `id` until none are left; returns #tasks run
  int64_t run_tasks(uint64_t id) {
    const Desc& d = desc_[id & 1];
    void* ctx = d.ctx.load(std::memory_order_relaxed);
    auto fn = d.fn.load(std::memory_order_relaxed);
    const int64_t n = d.ntasks.load(std::memory_order_relaxed);
    int64_t cnt = 0;
    uint64_t s = job_.load(std::memory_order_acquire);
    for (;;) {
      if ((s >> 32) != id || (int64_t)(s & 0xffffffffu) >= n) break;
      if (job_.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
        // CAS succeeded while job `id` was still current, so ctx/fn/n are its own
        try {
          fn(ctx, (int64_t)(s & 0xffffffffu));
        } catch (...) {
          std::lock_guard<std::mutex> lk(err_m_);
          if (!error_) error_ = std::current_exception();
        }
        ++cnt;
        s = job_.load(std::memory_order_acquire);
      }
    }
    return cnt;
  }

  void worker_loop() {
    uint64_t seen = job_.load(std::memory_order_acquire) >> 32;
    for (;;) {
      auto t0 = std::chrono::steady_clock::now();
      int spins = 0;
      uint64_t cur;
      while ((cur = job_.load(std::memory_order_acquire) >> 32) == seen) {
        if (stop_.load(std::memory_order_relaxed)) return;
        cpu_relax();
        if (++spins >= 128) {
          spins = 0;
          auto el = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
          if (el > kSpinMicroseconds) {
            std::unique_lock<std::mutex> lk(m_);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            while ((job_.load(std::memory_order_seq_cst) >> 32) == seen && !stop_.load())
              cv_.wait(lk);
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            t0 = std::chrono::steady_clock::now();
          }
        }
      }
      if (stop_.load()) return;
      seen = cur;
      const int64_t mine = run_tasks(cur);
      if (mine) done_.fetch_add(mine, std::memory_order_acq_rel);
    }
  }

  void stop_workers() {
    if (workers_.empty()) return;
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_.store(true);
      cv_.notify_all();
    }
    for (auto& t : workers_) t.join();
    workers_.clear();
  }

  static constexpr long kSpinMicroseconds = 300;

  const long pid_;  // process that created the workers
  int nthreads_ = 0;
  std::vector<std::thread> workers_;
  alignas(128) std::atomic<uint64_t> job_{0};
  alignas(128) std::atomic<int64_t> done_{0};
  alignas(128) std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  Desc desc_[2];
  std::mutex m_;
  std::mutex call_m_;  // one parallel region (or resize) at a time
  std::condition_variable cv_;
  std::mutex err_m_;
  std::exception_ptr error_;  // first exception of the current region
};

// The process-wide pool.  It is created on first use and never destroyed: at
// interpreter exit its idle workers simply end with the process, and in a child
// created by fork() the inherited pool (whose workers do not exist there) is
// abandoned rather than joined, and replaced by a new one.
inline ThreadPool& global_pool() {
  static std::atomic<ThreadPool*> current{nullptr};
  const long pid = process_id();
  ThreadPool* p = current.load(std::memory_order_acquire);
  while (p == nullptr || p->owner_pid() != pid) {
    ThreadPool* fresh = new ThreadPool(1);
    if (current.compare_exchange_strong(p, fresh, std::memory_order_acq_rel)) return *fresh;
    delete fresh;  // another thread installed a pool first; re-check it
  }
  return *p;
}

}  // namespace stalign
