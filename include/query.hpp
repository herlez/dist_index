#pragma once
#include <mpi.h>

#include <iostream>
#include <tdc/uint/uint40.hpp>

namespace alx::dist {

// A single query for the left or right border of the current backwards search
struct rank_query {
  rank_query() {
    m_pos_in_pattern = 0;
    m_border = 0;
    m_id = 0;
  }

  rank_query([[maybe_unused]] std::string const &pattern, size_t pattern_length, size_t border, size_t id) {
    std::memcpy(reinterpret_cast<char *>(m_pattern), pattern.c_str(), pattern.size());
    m_pos_in_pattern = pattern_length;
    m_border = border;
    m_id = id;
  }

  uint32_t m_id;
  unsigned char m_pattern[30];
  uint8_t m_pos_in_pattern;
  tdc::uint40_t m_border;

  unsigned char cur_char() const {
    return m_pattern[m_pos_in_pattern];
  }

  friend std::ostream &operator<<(std::ostream &os, const rank_query query) {
    return os << "id=" << query.m_id << "[" << (int)query.m_pos_in_pattern << "]=" << std::string(query.m_pattern, query.m_pattern + query.m_pos_in_pattern) << " border=" << query.m_border;
  }

  static MPI_Datatype mpi_type() {
    // Custom mpi data type for queries
    MPI_Aint displacements[4] = {offsetof(rank_query, m_id), offsetof(rank_query, m_pattern), offsetof(rank_query, m_pos_in_pattern), offsetof(rank_query, m_border)};
    int block_lengths[4] = {1, 30, 1, 5};
    MPI_Datatype types[4] = {MPI_UINT32_T, MPI_UNSIGNED_CHAR, MPI_UINT8_T, MPI_UNSIGNED_CHAR};

    MPI_Datatype custom_dt;

    // 2- Create the type, and commit it
    MPI_Type_create_struct(4, block_lengths, displacements, types, &custom_dt);
    return custom_dt;
  }
};

}  // namespace alx
