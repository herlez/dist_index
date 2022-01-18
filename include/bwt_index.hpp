#pragma once

#include <assert.h>
#include <mpi.h>

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
  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(m_bwt->cbegin(), m_bwt->cend(), 256));
  std::unique_ptr<wm_type> m_bwt_wm;
  alx::rank_query left_query;
  alx::rank_query right_query;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) {
    m_bwt_wm = std::make_unique<wm_type>(m_bwt->cbegin(), m_bwt->cend(), 256);
  }

  void rank(size_t local_pos, unsigned char c) {
    // Check if we are finished
    bool local_finished = !left_query.outstanding && !right_query.outstanding;
    bool global_finished;
    MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    // Open Window to write queries
    MPI_Win window;
    MPI_Win_create(&left_query, sizeof(left_query), sizeof(left_query), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    while (!global_finished) {
      if (left_query.outstanding && m_bwt->start_index() <= left_query.border.u64() && left_query.border.u64() < m_bwt->end_index()) {
        unsigned char c = left_query.cur_char();
        tdc::uint40_t left_border = left_query.border;

        // answer query
        size_t local_left_border = left_query.border.u64() - m_bwt->start_index();
        size_t global_rank = m_bwt->prev_occ(c) + local_rank(local_left_border, c);

        // prepare query for sending
        left_query.pos_in_pattern--;
        left_query.border = global_rank;

        // send query
        if (left_query.pos_in_pattern == 0) {
          // send to root and finish
        } else {
        }
        left_query.outstanding = false;
      }
      if (right_query.outstanding) {
        // answer query
        // send query
      }
      local_finished = !left_query.outstanding && !right_query.outstanding;
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    // Close Window to write queries
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);
  }

  size_t local_rank(size_t local_pos, unsigned char c) const {
    return m_bwt_wm->rank(local_pos + 1, c);
  }

  size_t text_size() const {
    return m_bwt->global_size();
  }
};
}  // namespace alx
