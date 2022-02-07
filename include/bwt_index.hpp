#pragma once

#include <assert.h>
#include <mpi.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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
  std::array<alx::rank_query, 2> m_queries;
  int m_num_outstanding_queries;
  std::vector<alx::rank_query> m_finished_queries;
  MPI_Datatype m_query_type = alx::rank_query::mpi_type();

  // MPI_Win window;

 public:
  bwt_index() : m_bwt(nullptr) { MPI_Type_commit(&m_query_type); }

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) { MPI_Type_commit(&m_query_type); }

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }

    // Answer single queries
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < patterns_size; ++i) {
      auto const& pattern = patterns[i];
      occ_single(pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(m_finished_queries[0].pos_in_pattern == 0 && m_finished_queries[1].pos_in_pattern == 0);
        size_t left, right;
        std::tie(left, right) = std::minmax(m_finished_queries[0].border.u64(), m_finished_queries[1].border.u64());
        results.push_back(right - left);
      }
    }
    return results;
  }

  void occ_single(std::string const& pattern) {
    initialize_query(pattern);
    alx::io::alxout << '\n'
                    << m_queries << '\n';

    // Loop : Send and Calculate until finished
    bool global_finished = false;
    while (!global_finished) {
      // Send queries with Alltoallv
      {
        std::array<alx::rank_query, 2> buffer_queries;

        // Sort queries by (are they finished?, border)
        std::sort(m_queries.begin(), m_queries.begin() + m_num_outstanding_queries, [](auto const& left, auto const& right) {
          if (left.pos_in_pattern == 0 && right.pos_in_pattern != 0) return true;
          return (left.border < right.border);
        });

        // Define my counts for sending (how many integers do I send to each process?)
        std::vector<int> counts_send(alx::mpi::world_size());
        for (int i = 0; i < m_num_outstanding_queries; ++i) {
          if (m_queries[i].pos_in_pattern != 0) {
            int target_pe = get_target_pe(m_queries[i].border.u64());
            ++counts_send[target_pe];
          } else if (m_queries[i].pos_in_pattern == 0) {
            // Send back to root
            ++counts_send[0];
          }
        }
        alx::io::alxout << "#SEND: " << counts_send << '\n';

        // Define my displacements for sending (where is located in the buffer each message to send?)
        std::vector<int> displacements_send(alx::mpi::world_size());
        std::exclusive_scan(counts_send.begin(), counts_send.end(), displacements_send.begin(), 0);
        alx::io::alxout << "DSEND: " << displacements_send << '\n';

        // Define my counts for receiving (how many integers do I receive from each process?)
        std::vector<int> counts_recv(alx::mpi::world_size());
        MPI_Alltoall(counts_send.data(), 1, MPI_INT, counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
        alx::io::alxout << "#RECV: " << counts_recv << '\n';

        // Define my displacements for reception (where to store in buffer each message received?)
        std::vector<int> displacements_recv(alx::mpi::world_size());
        std::exclusive_scan(counts_recv.begin(), counts_recv.end(), displacements_recv.begin(), 0);
        alx::io::alxout << "DRECV: " << displacements_recv << '\n';

        MPI_Alltoallv(m_queries.data(), counts_send.data(), displacements_send.data(), m_query_type, buffer_queries.data(), counts_recv.data(), displacements_recv.data(), m_query_type, MPI_COMM_WORLD);
        m_queries = buffer_queries;
        m_num_outstanding_queries = displacements_recv.back() + counts_recv.back();
        update_completed_queries();

        alx::io::alxout << m_queries << '\n';
      }

      // Calculate
      {
        for (int i = 0; i < m_num_outstanding_queries; ++i) {
          auto& query = m_queries[i];

          query.pos_in_pattern--;
          size_t next_border = m_bwt->next_border(query.border.u64(), query.cur_char());
          query.border = next_border;
          alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
        }
      }

      bool local_finished = (m_num_outstanding_queries == 0);
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
  }

  void initialize_query(std::string const& pattern) {
    m_finished_queries.clear();

    if (alx::mpi::my_rank() == 0) {
      m_num_outstanding_queries = sizeof(m_queries) / sizeof(m_queries[0]);
      for (int i = 0; i < m_num_outstanding_queries; ++i) {
        alx::rank_query& query = m_queries[i];
        query.outstanding = true;
        std::memcpy(reinterpret_cast<char*>(query.pattern), pattern.c_str(), pattern.size());
        query.pos_in_pattern = pattern.size();
        query.border = (i % 2 == 0) ? 0 : m_bwt->global_size();
        alx::io::alxout << "Initialized query " << i << ": " << query << "\n";
      }
    } else {
      m_num_outstanding_queries = 0;
    }
  }

  void update_completed_queries() {
    // Completed queries are
    if (alx::mpi::my_rank() == 0) {
      for (int i = 0; i < m_num_outstanding_queries; ++i) {
        if (m_queries[i].pos_in_pattern == 0) {
          m_finished_queries.push_back(m_queries[i]);

          // Delete completed query by swaping with last element
          m_queries[i] = m_queries.back();
          --m_num_outstanding_queries;
          --i;  // Look at this element again
        }
      }
    }
  }

  int get_target_pe(size_t border) {
    return std::get<0>(alx::io::locate_bwt_slice(border, m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
  }
};

}  // namespace alx
