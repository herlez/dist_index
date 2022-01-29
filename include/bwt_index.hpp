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

  alx::rank_query m_left_query;
  alx::rank_query m_right_query;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) {}

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }

    for (auto const& pattern : patterns) {
      occ_single(pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(m_left_query.pos_in_pattern == 0 && m_left_query.pos_in_pattern == 0);
        results.push_back(m_right_query.border.u64() - m_left_query.border.u64());
      }
    }

    return results;
  }

  void occ_single(std::string const& pattern) {
    // Open Window to write first query
    MPI_Win left_window;
    MPI_Win_create(&m_left_query, sizeof(m_left_query), sizeof(m_left_query), MPI_INFO_NULL, MPI_COMM_WORLD, &left_window);
    MPI_Win_fence(0, left_window);

    // Open Window to write first query
    MPI_Win right_window;
    MPI_Win_create(&m_right_query, sizeof(m_right_query), sizeof(m_right_query), MPI_INFO_NULL, MPI_COMM_WORLD, &right_window);
    MPI_Win_fence(0, right_window);

    if (alx::mpi::my_rank() == 0) {
      {
        m_left_query.outstanding = true;
        std::strcpy(reinterpret_cast<char*>(m_left_query.pattern), pattern.c_str());
        m_left_query.pos_in_pattern = pattern.size();
        m_left_query.border = 0;
        // Send left query to correct PE.
        int target_pe = std::get<0>(alx::io::locate_slice(m_left_query.border.u64(), m_bwt->global_size(), m_bwt->world_size()));

        if (target_pe != m_bwt->world_rank()) {
          MPI_Put(&m_left_query, sizeof(m_left_query), MPI_CHAR, target_pe, 0, sizeof(m_left_query), MPI_CHAR, left_window);
          m_left_query.outstanding = false;
        }
        alx::io::alxout << "Initialized first left query. " << m_left_query << "\n";
        alx::io::alxout << "Send to " << target_pe << "\n";
      }

      {
        m_right_query.outstanding = true;
        std::strcpy(reinterpret_cast<char*>(m_right_query.pattern), pattern.c_str());
        m_right_query.pos_in_pattern = pattern.size();
        m_right_query.border = m_bwt->global_size();
        // Send right query to correct PE.
        int target_pe = std::get<0>(alx::io::locate_slice(m_right_query.border.u64(), m_bwt->global_size(), m_bwt->world_size()));
        if (target_pe != m_bwt->world_rank()) {
          MPI_Put(&m_right_query, sizeof(m_right_query), MPI_CHAR, target_pe, 0, sizeof(m_right_query), MPI_CHAR, right_window);
          m_right_query.outstanding = false;
        }
        alx::io::alxout << "Initialized first right query. " << m_right_query << "\n";
        alx::io::alxout << "Send to " << target_pe << "\n";
      }
    }

    MPI_Win_fence(0, left_window);
    MPI_Win_free(&left_window);
    MPI_Win_fence(0, right_window);
    MPI_Win_free(&right_window);

    occ_single_distribute();
  }

  void occ_single_distribute() {
    // Check if we are finished
    bool local_finished = !m_left_query.outstanding && !m_right_query.outstanding;
    bool global_finished;
    MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    alx::io::alxout << "Done? " << global_finished << "\n";

    // Open Window to write queries
    MPI_Win left_window;
    MPI_Win_create(&m_left_query, sizeof(m_left_query), sizeof(m_left_query), MPI_INFO_NULL, MPI_COMM_WORLD, &left_window);

    // Open Window to write queries
    MPI_Win right_window;
    MPI_Win_create(&m_right_query, sizeof(m_right_query), sizeof(m_right_query), MPI_INFO_NULL, MPI_COMM_WORLD, &right_window);

    while (!global_finished) {
      if (m_left_query.outstanding) {
        alx::io::alxout << "Answering left:" << m_left_query << "\n";
        assert(m_bwt->start_index() <= m_left_query.border.u64() && m_left_query.border.u64() < m_bwt->end_index());

        // answer query
        m_left_query.pos_in_pattern--;
        size_t global_rank = m_bwt->global_rank(m_left_query.border.u64(), m_left_query.cur_char());
        m_left_query.border = global_rank;

        alx::io::alxout << "Calculated left:" << m_left_query << "\n";

        // send query
        if (m_left_query.pos_in_pattern == 0) {
          // send to root and finish
          m_left_query.outstanding = false;
          alx::io::alxout << "Sending left to root.\n";
          if (alx::mpi::my_rank() != 0) {
            MPI_Put(&m_left_query, sizeof(m_left_query), MPI_CHAR, 0, 0, sizeof(m_left_query), MPI_CHAR, left_window);
          }
        } else {
          // send to next processor
          int target_pe = std::get<0>(alx::io::locate_slice(global_rank, m_bwt->global_size(), m_bwt->world_size()));
          alx::io::alxout << "Sending left to " << target_pe << "\n";
          if (alx::mpi::my_rank() != target_pe) {
            MPI_Put(&m_left_query, sizeof(m_left_query), MPI_CHAR, target_pe, 0, sizeof(m_left_query), MPI_CHAR, left_window);
            m_left_query.outstanding = false;
          }
        }
      }
      if (m_right_query.outstanding) {
        assert(m_bwt->start_index() <= m_right_query.border.u64() && m_right_query.border.u64() < m_bwt->end_index());

        // answer query
        m_right_query.pos_in_pattern--;
        size_t global_rank = m_bwt->global_rank(m_right_query.border.u64(), m_right_query.cur_char());
        m_right_query.border = global_rank;

        // send query
        if (m_right_query.pos_in_pattern == 0) {
          // send to root and finish
          m_right_query.outstanding = false;
          alx::io::alxout << "Sending left to root.\n";
          if (alx::mpi::my_rank() != 0) {
            MPI_Put(&m_right_query, sizeof(m_right_query), MPI_CHAR, 0, 0, sizeof(m_right_query), MPI_CHAR, right_window);
          }
        } else {
          // send to next processor
          int target_pe = std::get<0>(alx::io::locate_slice(global_rank, m_bwt->global_size(), m_bwt->world_size()));
          alx::io::alxout << "Sending right to " << target_pe << "\n";
          if (alx::mpi::my_rank() != target_pe) {
            MPI_Put(&m_right_query, sizeof(m_right_query), MPI_CHAR, target_pe, 0, sizeof(m_right_query), MPI_CHAR, right_window);
            m_right_query.outstanding = false;
          }
        }
      }
      local_finished = !m_left_query.outstanding && !m_right_query.outstanding;
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
      alx::io::alxout << "Done? " << global_finished << "\n";
    }

    // Close Window to write queries
    MPI_Win_fence(0, right_window);
    MPI_Win_fence(0, left_window);
    MPI_Win_free(&left_window);
    MPI_Win_free(&right_window);
  }

  /*
  size_t get_count() {
    assert(f.pos_in_pattern == 0 == m_right_query.pos_in_pattern);
    return m_right_query.border.u64() - m_left_query.border.u64();
  }

  void occ_distributed() {
    // Check if we are finished
    bool local_finished = !m_left_query.outstanding && !m_right_query.outstanding;
    bool global_finished;
    MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    // Open Window to write queries
    MPI_Win window;
    MPI_Win_create(&m_left_query, sizeof(m_left_query), sizeof(m_left_query), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    while (!global_finished) {
      if (m_left_query.outstanding && m_bwt->start_index() <= m_left_query.border.u64() && m_left_query.border.u64() < m_bwt->end_index()) {
        unsigned char c = m_left_query.cur_char();
        tdc::uint40_t left_border = m_left_query.border;

        // answer query
        size_t local_left_border = left_border.u64() - m_bwt->start_index();
        size_t global_rank = m_bwt->prev_occ(c) + local_rank(local_left_border, c);

        // prepare query for sending
        m_left_query.pos_in_pattern--;
        m_left_query.border = global_rank;

        // send query
        if (m_left_query.pos_in_pattern == 0) {
          // send to root and finish
          m_left_query.outstanding = false;
          MPI_Put(&m_left_query, sizeof(m_left_query), MPI_CHAR, 0, 0, sizeof(m_left_query), MPI_CHAR, window);
        } else {
          // send to next processor
          int target_pe = std::get<0>(alx::io::locate_slice(global_rank, m_bwt->global_size(), m_bwt->world_size()));
          MPI_Put(&m_left_query, sizeof(m_left_query), MPI_CHAR, target_pe, 0, sizeof(m_left_query), MPI_CHAR, window);
        }
        // if we send query to other pe, we don't have to answer query in next rotation
        int target_pe = std::get<0>(alx::io::locate_slice(global_rank, m_bwt->global_size(), m_bwt->world_size()));
        if (target_pe != m_bwt->world_rank()) {
          m_left_query.outstanding = false;
        }
      }
      if (m_right_query.outstanding) {
        // answer query
        // send query
      }
      local_finished = !m_left_query.outstanding && !m_right_query.outstanding;
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    // Close Window to write queries
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);
  }
  */
};

}  // namespace alx
