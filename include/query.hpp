#pragma once
#include <iostream>
#include <tdc/uint/uint40.hpp>
#include <mpi.h>

namespace alx {

// A single query for the left or right border of the current backwards search
struct rank_query {
  rank_query() {
    pos_in_pattern = 0;
    border = 0;
  }
  uint32_t id;
  unsigned char pattern[30];
  uint8_t pos_in_pattern;

  tdc::uint40_t border;

  unsigned char cur_char() const {
    return pattern[pos_in_pattern];
  }

  friend std::ostream &operator<<(std::ostream &os, const alx::rank_query query) {
    return os << "id=" << query.id << "[" << (int) query.pos_in_pattern << "]=" << std::string(query.pattern, query.pattern + query.pos_in_pattern) << " border=" << query.border;
  }

  static MPI_Datatype mpi_type() {
    //Custom mpi data type for queries
        MPI_Aint displacements[4] = {offsetof(rank_query, id), offsetof(rank_query, pattern), offsetof(rank_query, pos_in_pattern), offsetof(rank_query, border)};
        int block_lengths[4] = {1, 30, 1, 5};
        MPI_Datatype types[4] = {MPI_UINT32_T, MPI_UNSIGNED_CHAR, MPI_UINT8_T, MPI_UNSIGNED_CHAR};
        
        MPI_Datatype custom_dt;

        // 2- Create the type, and commit it
        MPI_Type_create_struct(4, block_lengths, displacements, types, &custom_dt);
        return custom_dt;
  }

};

}  // namespace alx
