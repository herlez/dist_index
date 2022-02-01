#pragma once

#include <assert.h>
#include <mpi.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <wavelet_tree/wavelet_tree.hpp>

#include "bwt.hpp"
#include "query.hpp"
#include "util/io.hpp"

namespace alx {

template <typename t_word = tdc::uint40_t>
class bwt_index {
 private:
  alx::bwt const* m_bwt;
  alx::rank_query m_queries[2];
  MPI_Win window;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) {}

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    MPI_Win_create(&m_queries, sizeof(m_queries), sizeof(m_queries[0]), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }

    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < patterns_size; ++i) {
      auto const& pattern = patterns[i];
      occ_single(pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(m_queries[0].pos_in_pattern == 0 && m_queries[1].pos_in_pattern == 0);
        results.push_back(m_queries[1].border.u64() - m_queries[0].border.u64());
      }
    }
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    return results;
  }

  void occ_single(std::string const& pattern) {
    

    if (alx::mpi::my_rank() == 0) {
      for (size_t i = 0; i < sizeof(m_queries) / sizeof(m_queries[0]); ++i) {
        alx::rank_query& query = m_queries[i];
        query.outstanding = true;
        std::memcpy(reinterpret_cast<char*>(query.pattern), pattern.c_str(), pattern.size());
        query.pos_in_pattern = pattern.size();
        query.border = (i == 0) ? 0 : m_bwt->global_size();

        int target_pe = std::get<0>(alx::io::locate_bwt_slice(query.border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
        if (target_pe != m_bwt->world_rank()) {
          MPI_Put(m_queries + i, sizeof(alx::rank_query), MPI_CHAR, target_pe, i, sizeof(alx::rank_query), MPI_CHAR, window);
        }
        alx::io::alxout << "\nInitialized query " << i << ": " << query << "\n";
        alx::io::alxout << "Send to " << target_pe << "\n";
      }
    }
    MPI_Win_fence(0, window);
    update_outstanding_status();
    occ_single_distribute();
  }

  void occ_single_distribute() {
    // Check if we are finished
    bool local_finished = !m_queries[0].outstanding && !m_queries[1].outstanding;
    bool global_finished;
    MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(global_finished) {alx::io::alxout << "Done.\n";}

  
    while (!global_finished) {
      MPI_Win_fence(0, window);
      for (size_t i = 0; i < sizeof(m_queries) / sizeof(m_queries[0]); ++i) {
        alx::rank_query& query = m_queries[i];
        if (query.outstanding) {
          alx::io::alxout << "Answering query[" << i << "]: " << query << "\n";
          assert(m_bwt->start_index() <= query.border.u64() && query.border.u64() < m_bwt->end_index());

          // answer query
          query.pos_in_pattern--;
          size_t next_border = m_bwt->next_border(query.border.u64(), query.cur_char());
          query.border = next_border;

          alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";

          // send query
          if (query.pos_in_pattern == 0) {
            // send to root and finish
            query.outstanding = false;
            alx::io::alxout << "Sending query[" << i << "] to root.\n";
            if (alx::mpi::my_rank() != 0) {
              MPI_Put(&query, sizeof(query), MPI_CHAR, 0, i, sizeof(query), MPI_CHAR, window);
            }
          } else {
            // send to next processor
            int target_pe = std::get<0>(alx::io::locate_bwt_slice(next_border, m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
            alx::io::alxout << "Sending query[" << i << "] to " << target_pe << ".\n";
            if (alx::mpi::my_rank() != target_pe) {
              MPI_Put(&query, sizeof(query), MPI_CHAR, target_pe, i, sizeof(query), MPI_CHAR, window);
              
            }
          }
        }
      }
      MPI_Win_fence(0, window);
      update_outstanding_status();


      alx::io::alxout << "Q0:" << m_queries[0] << "\n";
      alx::io::alxout << "Q1:" << m_queries[1] << "\n";

      local_finished = !m_queries[0].outstanding && !m_queries[1].outstanding;
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
      if(global_finished) {alx::io::alxout << "Done.\n";}
    }
  }

  void update_outstanding_status() {
    for(auto& query : m_queries) {
      int target_pe = std::get<0>(alx::io::locate_bwt_slice(query.border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
        if (target_pe != m_bwt->world_rank()) {
          query.outstanding = false;
        }
    }
  }

};



}  // namespace alx
