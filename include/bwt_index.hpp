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
  std::vector<alx::rank_query> m_queries;
  std::vector<alx::rank_query> m_finished_queries;
  MPI_Datatype m_query_type = alx::rank_query::mpi_type();

 public:
  bwt_index() : m_bwt(nullptr) { MPI_Type_commit(&m_query_type); }

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) { MPI_Type_commit(&m_query_type); }

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }
    // Answer queries in batch
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    alx::io::alxout << "#Patterns=: " << patterns_size << "\n";
    alx::io::alxout << patterns;

    m_finished_queries.clear();
    occ(patterns);

    if (alx::mpi::my_rank() == 0) {
      assert(m_finished_queries.size() == 2 * patterns.size());
      std::sort(m_finished_queries.begin(), m_finished_queries.end(), [](auto const& left, auto const& right) {
        return left.id < right.id;
      });

      for (size_t i = 0; i < m_finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(m_finished_queries[i].border.u64(), m_finished_queries[i + 1].border.u64());
        alx::io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  std::vector<size_t> occ_one_by_one(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }

    // Answer single queries
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    alx::io::alxout << "#Patterns=: " << patterns_size << "\n";

    for (size_t i = 0; i < patterns_size; ++i) {
      std::vector<std::string> single_pattern;

      if (mpi::my_rank() == 0) {
        single_pattern.push_back(patterns[i]);
        alx::io::alxout << "Queries: " << single_pattern << "\n";
      }

      m_finished_queries.clear();
      occ(single_pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(m_finished_queries[0].pos_in_pattern == 0 && m_finished_queries[1].pos_in_pattern == 0);
        assert(m_finished_queries.size() == 2 * single_pattern.size());
        size_t left, right;
        std::tie(left, right) = std::minmax(m_finished_queries[0].border.u64(), m_finished_queries[1].border.u64());
        alx::io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  void occ(std::vector<std::string> const& patterns) {
    initialize_query(patterns);
    alx::io::alxout << '\n'
                    << m_queries << '\n';

    // Loop : Send and Calculate until finished
    bool global_finished = false;
    while (!global_finished) {
      // Send queries with Alltoallv
      {
        // Sort queries by (are they finished?, border)
        std::sort(m_queries.begin(), m_queries.end(), [](auto const& left, auto const& right) {
          if (left.pos_in_pattern == 0 && right.pos_in_pattern != 0) return true;
          return (left.border < right.border);
        });

        // Define my counts for sending (how many integers do I send to each process?)
        std::vector<int> counts_send(alx::mpi::world_size());
        for (size_t i = 0; i < m_queries.size(); ++i) {
          int target_pe = get_target_pe(m_queries[i]);
          ++counts_send[target_pe];
        }
        alx::io::alxout << "#SEND: " << counts_send << '\n';

        // Define my displacements for sending (where is located in the buffer each message to send?)
        std::vector<int> displacements_send(alx::mpi::world_size());
        std::exclusive_scan(counts_send.begin(), counts_send.end(), displacements_send.begin(), 0);
        alx::io::alxout << "DSEND: " << displacements_send << '\n';

        // Define my counts for receiving (how many integers do I receive from each process?)
        std::vector<int> counts_recv(alx::mpi::world_size());
        alx::io::alxout << "rect with size: " << counts_recv.size() << '\n';
        MPI_Alltoall(counts_send.data(), 1, MPI_INT, counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
        alx::io::alxout << "rect with size: " << counts_recv.size() << '\n';
        alx::io::alxout << "#RECV: " << counts_recv << '\n';

        // Define my displacements for reception (where to store in buffer each message received?)
        std::vector<int> displacements_recv(alx::mpi::world_size());
        std::exclusive_scan(counts_recv.begin(), counts_recv.end(), displacements_recv.begin(), 0);
        alx::io::alxout << "DRECV: " << displacements_recv << '\n';

        int num_incoming_queries = displacements_recv.back() + counts_recv.back();
        std::vector<alx::rank_query> buffer_queries(num_incoming_queries);

        MPI_Alltoallv(m_queries.data(), counts_send.data(), displacements_send.data(), m_query_type, buffer_queries.data(), counts_recv.data(), displacements_recv.data(), m_query_type, MPI_COMM_WORLD);
        m_queries = buffer_queries;

        alx::io::alxout << m_queries << '\n';
      }

      // Calculate
      {
        for (size_t i = 0; i < m_queries.size(); ++i) {
          auto& query = m_queries[i];
          /*
          while ((get_target_pe(query) == alx::mpi::my_rank()) && query.pos_in_pattern != 0) {
            query.pos_in_pattern--;
            query.border = m_bwt->next_border(query.border.u64(), query.cur_char());
            alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
          }*/

          if (query.pos_in_pattern != 0) {
            query.pos_in_pattern--;
            query.border = m_bwt->next_border(query.border.u64(), query.cur_char());
            alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
          }
        }
      }
      update_completed_queries();

      bool local_finished = m_queries.empty();
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
  }

  void initialize_query(std::vector<std::string> const& patterns) {
    alx::io::alxout << "Initializing queries..\n";
    m_queries.resize(patterns.size() * 2);
    m_finished_queries.clear();

    if (alx::mpi::my_rank() == 0) {
      for (size_t i = 0; i < m_queries.size(); ++i) {
        alx::rank_query& query = m_queries[i];
        std::string const& pattern = patterns[i / 2];

        query.id = i / 2;
        std::memcpy(reinterpret_cast<char*>(query.pattern), pattern.c_str(), pattern.size());
        query.pos_in_pattern = pattern.size();
        query.border = (i % 2 == 0) ? 0 : m_bwt->global_size();
        alx::io::alxout << "Initialized query " << i << ": " << query << "\n";
      }
    } else {
      m_queries.resize(0);
    }
  }

  void update_completed_queries() {
    // Completed queries are
    if (alx::mpi::my_rank() == 0) {
      for (size_t i = 0; i < m_queries.size(); ++i) {
        if (m_queries[i].pos_in_pattern == 0) {
          m_finished_queries.push_back(m_queries[i]);

          // Delete completed query by swaping with last element
          m_queries[i] = m_queries.back();
          m_queries.pop_back();
          --i;  // Look at this element again
        }
      }
    }
  }

  int get_target_pe(rank_query const& query) {
    if (query.pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(alx::io::locate_bwt_slice(query.border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
    }
  }
};

}  // namespace alx