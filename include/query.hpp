#pragma once
#include <iostream>
#include <tdc/uint/uint40.hpp>

namespace alx {

// A single query for the left or right border of the current backwards search
struct rank_query {
  rank_query() {
    outstanding = false;
    pos_in_pattern = 0;
    border = 0;
  }

  bool outstanding;
  unsigned char pattern[30];
  uint8_t pos_in_pattern;

  tdc::uint40_t border;

  unsigned char cur_char() const {
    return pattern[pos_in_pattern];
  }

  friend std::ostream &operator<<(std::ostream &os, const alx::rank_query query) {
    return os << "[" << (int) query.pos_in_pattern << "]=" << std::string(query.pattern, query.pattern + query.pos_in_pattern) << " border=" << query.border;
  }
};

}  // namespace alx
