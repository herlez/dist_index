#pragma once

#include <assert.h>
#include <mpi.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <wavelet_tree/wavelet_tree.hpp>

#include "util/io.hpp"

namespace alx::dist {

class bwt {
 private:
  size_t m_global_size;  // size of bwt
  size_t m_start_index;  // start of bwt slice
  size_t m_end_index;    // end of bwt slice (exclusive)
  int m_world_size;      // number of PEs
  int m_world_rank;      // PE number

  alx::ustring m_last_row;                               // last row bwt matrix
  std::array<size_t, 256> m_exclusive_prefix_histogram;  // histogram of text of previous PEs
  std::array<size_t, 256> m_first_row_starts;            // positions where the character runs start in global F

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(alx::ustring::const_iterator{}, alx::ustring::const_iterator{}, 256));
  std::unique_ptr<wm_type> m_wm;  // wavelet tree to support rank

 public:
  bwt() : m_global_size{0}, m_start_index{0}, m_end_index{0} {}

  // Load partial bwt from bwt file.
  bwt(std::filesystem::path const& last_row_path) {
    MPI_Comm_size(MPI_COMM_WORLD, &m_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_world_rank);

    // If file does not exist, return empty string.
    if (!std::filesystem::exists(last_row_path)) {
      io::alxout << last_row_path << " does not exist.";
      return;
    }
    // Read last row.
    {
      m_global_size = std::filesystem::file_size(last_row_path);
      std::tie(m_start_index, m_end_index) = slice_indexes(m_global_size, m_world_rank, m_world_size);

      MPI_File handle;
      if (MPI_File_open(MPI_COMM_WORLD, last_row_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
        std::cout << "Failure in opening the file.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      m_last_row.resize(size());
      // Because MPI_File_read_at_all only supports reading MAX_INT characters we do more iterations
      size_t read_already = 0;
      bool read_finished = (read_already == size());
      while (!read_finished) {
        size_t read_next = std::min(size_t{1} << 30, size() - read_already);
        MPI_File_read_at_all(handle, m_start_index + read_already, m_last_row.data() + read_already, read_next, MPI_CHAR, MPI_STATUS_IGNORE);

        read_already += read_next;        
        if(read_already == size()) {
          read_finished = true;
        }
        MPI_Allreduce(MPI_IN_PLACE, &read_finished, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
      }
      MPI_File_close(&handle);
    }

    // Build local histogram. Use m_first_row_starts temporarily
    {
      m_exclusive_prefix_histogram.fill(0);
      m_first_row_starts.fill(0);
      for (char c : m_last_row) {
        ++m_first_row_starts[c];
      }
      /*for (size_t i = 0; i < m_first_row_starts.size(); ++i)
        std::cout << m_first_row_starts[i] << " ";
      std::cout << "\n";*/
      // Exclusive scan histogram
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Exscan(m_first_row_starts.data(), m_exclusive_prefix_histogram.data(), 256, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

      // std::cout << m_first_row_starts << "\n";
      // std::cout << m_exclusive_prefix_histogram << "\n";
      /*for (size_t i = 0; i < m_exclusive_prefix_histogram.size(); ++i)
        std::cout << m_exclusive_prefix_histogram[i] << " ";
      std::cout << "\n";*/
    }

    // Broadcast start of runs in first row
    // Last PE: Add exclusive prefix histogram to local histogram to get global histogram
    {
      if (m_world_rank == m_world_size - 1) {
        for (size_t i = 0; i < m_first_row_starts.size(); ++i) {
          m_first_row_starts[i] += m_exclusive_prefix_histogram[i];
        }
        // Exclusive scan histogram to get global starting positions
        std::exclusive_scan(m_first_row_starts.begin(), m_first_row_starts.end(), m_first_row_starts.begin(), size_t{0});
        // std::cout << m_first_row_starts << "\n";
        /*for (size_t i = 0; i < m_first_row_starts.size(); ++i)
          std::cout << m_first_row_starts[i] << " ";
        std::cout << "\n";*/
      }
      // Broadcast global starting positions
      MPI_Bcast(m_first_row_starts.data(), m_first_row_starts.size(), my_MPI_SIZE_T, m_world_size - 1, MPI_COMM_WORLD);
      /*for (size_t i = 0; i < m_exclusive_prefix_histogram.size(); ++i)
        std::cout << m_exclusive_prefix_histogram[i] << " ";
      std::cout << "\n";*/
    }
  }

  // Getter
  size_t global_size() const {
    return m_global_size;
  }
  size_t start_index() const {
    return m_start_index;
  }
  size_t end_index() const {
    return m_end_index;
  }
  int world_size() const {
    return m_world_size;
  }
  int world_rank() const {
    return m_world_rank;
  }
  std::array<size_t, 256> exclusive_prefix_histogram() const {
    return m_exclusive_prefix_histogram;
  }

  std::array<size_t, 256> first_row_starts() const {
    return m_first_row_starts;
  }
  ustring last_row() {
    return m_last_row;
  }
  alx::ustring::value_type access_bwt(size_t i) const {
    return m_last_row[i];
  }
  /*alx::ustring::value_type access_wm(size_t i) const {
    return m_wm->operator[](i);
  }*/

  // Return size of bwt slice.
  size_t size() const {
    return m_end_index - m_start_index;
  }

  size_t local_rank(size_t local_pos, unsigned char c) const {
    return m_wm->rank(local_pos, c);
  }

  size_t global_rank(size_t global_pos, unsigned char c) const {
    size_t slice;
    size_t local_pos;
    std::tie(slice, local_pos) = locate_bwt_slice(global_pos, m_global_size, m_world_size);
    assert(slice == m_world_rank);
    io::alxout << "Answering rank. global_pos=" << global_pos << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
    io::alxout << "Result: " << m_exclusive_prefix_histogram[c] + local_rank(local_pos, c) << "\n";
    return m_exclusive_prefix_histogram[c] + local_rank(local_pos, c);
  }

  size_t next_border(size_t global_pos, unsigned char c) const {
    return m_first_row_starts[c] + global_rank(global_pos, c);
  }

  void build_rank() {
    m_wm = std::make_unique<wm_type>(m_last_row.cbegin(), m_last_row.cend(), 256);
  }

  void free_bwt() {
    alx::ustring str;
    std::swap(m_last_row, str);
  }

  template <typename t_query>
  int get_target_pe(t_query const& query) const {
    if (query.m_pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(locate_bwt_slice(query.m_border.u64(), m_global_size, m_world_size));
    }
  }

  static std::tuple<size_t, size_t> locate_bwt_slice(size_t global_index, size_t global_size, size_t world_size) {
    return alx::dist::locate_slice(global_index, global_size, world_size);
  }
};
}  // namespace alx::dist
