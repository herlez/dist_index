#pragma once
#include <mpi.h>

#include <iostream>
#include <tdc/uint/uint40.hpp>

namespace alx::dist {

// A single query for the left or right border of the current backwards search
struct rank_query_information {
  rank_query_information() {
    m_pos_in_pattern = 0;
    m_border = 0;
    m_id = 0;
  }

  rank_query_information([[maybe_unused]] std::string const &pattern, size_t pattern_length, size_t border, size_t id) {
    m_pos_in_pattern = pattern_length;
    m_border = border;
    m_id = id;
  }

  uint32_t m_id;
  uint16_t m_pos_in_pattern;
  tdc::uint40_t m_border;

  friend std::ostream &operator<<(std::ostream &os, const rank_query_information query) {
    return os << "id=" << query.m_id << "[" << (int)query.m_pos_in_pattern << "] border=" << query.m_border;
  }

  static MPI_Datatype mpi_type() {
    // Custom mpi data type for queries
    MPI_Aint displacements[3] = {offsetof(rank_query_information, m_id), offsetof(rank_query_information, m_pos_in_pattern), offsetof(rank_query_information, m_border)};
    int block_lengths[3] = {1, 1, 5};
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT8_T, MPI_UNSIGNED_CHAR};

    MPI_Datatype custom_dt;

    // 2- Create the type, and commit it
    MPI_Type_create_struct(3, block_lengths, displacements, types, &custom_dt);
    return custom_dt;
  }
};

}  // namespace alx
