#pragma once

#include <malloc_count.h>

namespace alx::benchutil {

class spacer {
 private:
  double m_begin;

 public:
  spacer() : m_begin(malloc_count_current()) {}

  void reset() {
    m_begin = malloc_count_current();
  }

  int64_t get() {
    return malloc_count_current() - m_begin;
  }

  int64_t get_and_reset() {
    auto before = m_begin;  
    m_begin = malloc_count_current();
    return m_begin - before;
  }

  int64_t get_peak() {
    return malloc_count_peak() - m_begin;
  }

  void reset_peak() {
    malloc_count_reset_peak();
  }
};
}  // namespace alx::benchutil
