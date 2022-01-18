#pragma once

#include <assert.h>
#include <mpi.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "util/io.hpp"

namespace alx {

template <typename t_word = tdc::uint40_t>
class bwt_rle {
 private:
  size_t m_start_index;
  size_t m_end_index;
  size_t m_global_size;

  size_t m_primary_index;
  std::vector<t_word> m_run_starts;
  tdc::pred::Index<t_word> m_run_starts_pred;
  alx::ustring m_run_chars;
  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(m_run_chars.cbegin(), m_run_chars.cend(), 256));
  std::unique_ptr<wm_type> m_run_chars_wm;

  alx::ustring m_last_row;
  std::array<size_t, 256> m_prev_occ;

 public:
  bwt_rle() = default;

  // Load partial bwt from bwt and primary index file. //CONTINUE HERE
  bwt_rle(std::filesystem::path const& last_row_path, std::filesystem::path const& primary_index_path, int world_rank, int world_size) {
    // If file does not exist, return empty string
    if (!std::filesystem::exists(last_row_path)) {
      std::cerr << last_row_path << " does not exist.";
      return;
    }
    if (!std::filesystem::exists(primary_index_path)) {
      std::cerr << primary_index_path << " does not exist.";
      return;
    }

    // Read primary index
    {
      std::ifstream in(primary_index_path, std::ios::binary);
      in.read(reinterpret_cast<char*>(&m_primary_index), sizeof(m_primary_index));
    }
    // Read last row
    {
      std::ifstream in(last_row_path, std::ios::binary);
      in.seekg(0, std::ios::beg);
      std::streampos begin = in.tellg();
      in.seekg(0, std::ios::end);
      size_t size = in.tellg() - begin;
      std::tie(m_start_index, m_end_index) = alx::io::slice_indexes(size, world_rank, world_size);

      m_last_row.resize(size);
      in.seekg(m_start_index, std::ios::beg);
      in.read(reinterpret_cast<char*>(m_last_row.data()), m_end_index - m_start_index);
    }

    // Initial global prefix sums
    m_prev_occ.fill(0);

    // Build local
    std::array<size_t, 256> histogram;
    for (char c : m_last_row) {
      ++histogram[c];
    }
    // Exclusive scan histogram
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Exscan(histogram.data(), m_prev_occ.data(), 256, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
  }

  size_t rank(size_t index, unsigned char c) {
    //more complicated
    return m_prev_occ[c];
  }
  size_t local_rank(size_t local_pos, unsigned char c) {
    //More complicated
    return m_run_chars_wm->rank(local_pos + 1, c);
  }
};
}  // namespace alx
