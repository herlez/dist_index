#pragma once
#include <iostream>
#include <tdc/uint/uint40.hpp>
#include <mpi.h>

namespace alx {

// A single query for the left or right border of the current backwards search
struct rank_query_information {
  rank_query_information() {
    pos_in_pattern = 0;
    border = 0;
  }
  uint32_t id;
  uint16_t pos_in_pattern;
  tdc::uint40_t border;

  friend std::ostream &operator<<(std::ostream &os, const alx::rank_query_information query) {
    return os << "id=" << query.id << "[" << (int) query.pos_in_pattern << "] border=" << query.border;
  }

  static MPI_Datatype mpi_type() {
    //Custom mpi data type for queries
        MPI_Aint displacements[3] = {offsetof(rank_query_information, id), offsetof(rank_query_information, pos_in_pattern), offsetof(rank_query_information, border)};
        int block_lengths[3] = {1, 1, 5};
        MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT8_T, MPI_UNSIGNED_CHAR};
        
        MPI_Datatype custom_dt;

        // 2- Create the type, and commit it
        MPI_Type_create_struct(3, block_lengths, displacements, types, &custom_dt);
        return custom_dt;
  }

};

}  // namespace alx
