#pragma once

#include <chrono>

namespace alx::benchutil {

class timer {
 private:
  std::chrono::system_clock::time_point m_begin;

 public:
  timer() : m_begin(std::chrono::system_clock::now()) {}

  void reset() {
    m_begin = std::chrono::system_clock::now();
  }

  size_t get() const {
    auto const end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<
               std::chrono::milliseconds>(end - m_begin)
        .count();
  }

  size_t get_and_reset() {
    auto const time = get();
    reset();
    return time;
  }
};  // class Timer
}  // namespace sss::util