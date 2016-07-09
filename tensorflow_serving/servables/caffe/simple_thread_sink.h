#pragma once 

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class SimpleThreadSink {
 public:
  SimpleThreadSink();
  
  template<class F, class... Args>
  auto run(F&& f, Args&&... args) 
      -> typename std::result_of<F(Args...)>::type;
      
  ~SimpleThreadSink();
 private:
  // joinable worker
  std::thread worker_;
  // the task queue
  std::queue<std::function<void()>> tasks_;
  
  // synchronization
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};


//////////
// Implementation details follow. API users need not read.


// add new work item to the pool and block until its complete
template<class F, class... Args>
auto SimpleThreadSink::run(F&& f, Args&&... args) 
    -> typename std::result_of<F(Args...)>::type
{
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::packaged_task<return_type()>(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...));
      
  std::future<return_type> res = task.get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_.emplace([&task](){ (task)(); });
    condition_.notify_one();
  }
  return res.get();
}
