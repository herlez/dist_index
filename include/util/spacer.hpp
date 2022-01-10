#pragma once

#include <malloc_count.h>
#include <stack_count.h>

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

  int64_t get_peak() {
    return malloc_count_peak() - m_begin;
  }
};
}  // namespace sss::util
