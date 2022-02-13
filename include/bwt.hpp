#pragma once

#include <assert.h>
#include <mpi.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <wavelet_tree/wavelet_tree.hpp>

#include "util/io.hpp"

namespace alx {

class bwt {
 private:
  size_t m_global_size;  // size of bwt
  size_t m_start_index;  // start of bwt slice
  size_t m_end_index;    // end of bwt slice (exclusive)
  int m_world_size;      // number of PEs
  int m_world_rank;      // PE number

  alx::ustring m_last_row;                               // last row bwt matrix
  size_t m_primary_index;                                // index of implicit $ in last row
  std::array<size_t, 256> m_exclusive_prefix_histogram;  // histogram of text of previous PEs
  std::array<size_t, 256> m_first_row_starts;            // positions where the character runs start in F

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(alx::ustring::const_iterator{}, alx::ustring::const_iterator{}, 256));
  std::unique_ptr<wm_type> m_wm;  // wavelet tree to support rank

 public:
  bwt() : m_global_size{0}, m_start_index{0}, m_end_index{0} {}

  // Load partial bwt from bwt and primary index file.
  bwt(std::filesystem::path const& last_row_path, std::filesystem::path const& primary_index_path) {
    MPI_Comm_size(MPI_COMM_WORLD, &m_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_world_rank);

    // If file does not exist, return empty string.
    if (!std::filesystem::exists(last_row_path)) {
      alx::io::alxout << last_row_path << " does not exist.";
      return;
    }
    if (!std::filesystem::exists(primary_index_path)) {
      alx::io::alxout << primary_index_path << " does not exist.";
      return;
    }

    // Read primary index.
    {
      std::ifstream in(primary_index_path, std::ios::binary);
      in.read(reinterpret_cast<char*>(&m_primary_index), sizeof(m_primary_index));
    }
    // Read last row.
    {
      m_global_size = std::filesystem::file_size(last_row_path);
      std::tie(m_start_index, m_end_index) = alx::io::slice_indexes(m_global_size, m_world_rank, m_world_size);

      MPI_File handle;
      if (MPI_File_open(MPI_COMM_WORLD, last_row_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &handle) != MPI_SUCCESS)
      {
        std::cout << "Failure in opening the file.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      m_last_row.resize(size());
      MPI_File_read_at_all(handle, m_start_index, m_last_row.data(), size(), MPI_CHAR, MPI_STATUS_IGNORE);
      MPI_File_close(&handle);
    }

    // Build local histogram. Use m_first_row_starts temporarily
    {
      m_exclusive_prefix_histogram.fill(0);
      m_first_row_starts.fill(0);
      for (char c : m_last_row) {
        ++m_first_row_starts[c];
      }
      // Exclusive scan histogram
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Exscan(m_first_row_starts.data(), m_exclusive_prefix_histogram.data(), 256, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

      // alx::io::alxout << m_first_row_starts << "\n";
      // alx::io::alxout << m_exclusive_prefix_histogram << "\n";
    }

    // Broadcast start of runs in first row
    // Last PE: Add exclusive prefix histogram to local histogram to get global histogram
    {
      if (m_world_rank == m_world_size - 1) {
        for (size_t i = 0; i < m_first_row_starts.size(); ++i) {
          m_first_row_starts[i] += m_exclusive_prefix_histogram[i];
        }
      }
      // Exclusive scan histogram to get starting positions
      std::exclusive_scan(m_first_row_starts.begin(), m_first_row_starts.end(), m_first_row_starts.begin(), 0);
      MPI_Bcast(m_first_row_starts.data(), m_first_row_starts.size(), my_MPI_SIZE_T, m_world_size - 1, MPI_COMM_WORLD);

      alx::io::alxout << m_first_row_starts << "\n";
    }
  }

  // Calulate partial bwt from distributed suffix array and distributed text.
  template <typename t_text_container, typename t_sa_container>
  bwt(t_text_container const& text_slice, t_sa_container const& sa_slice, size_t text_size, int world_rank, int world_size) {
    m_world_size = world_size;
    m_world_rank = world_rank;
    std::tie(m_start_index, m_end_index) = alx::io::slice_indexes(text_size, world_rank, world_size);

    assert(text_slice.size() == sa_slice.size());

    m_last_row.reserve(text_slice.size());
    if (world_rank == 0) {
      m_last_row.push_back(text_slice.back());  // text[0] = imaginary $
    }

    // Open suffix array for mpi
    MPI_Win window;
    MPI_Win_create(text_slice.data(), text_slice.size() * sizeof(sa_slice.size_type), sizeof(sa_slice.size_type), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);
    m_primary_index = 0;
    for (size_t i{0}; i < sa_slice.size(); ++i) {
      if (sa_slice[i] == 0) [[unlikely]] {
        m_primary_index = m_start_index + i;
      } else {
        size_t requested_global_index = sa_slice[i] - 1;
        size_t target_rank;  // PE# in which the char lies
        size_t local_index;  // index in PE at which char lies
        std::tie(target_rank, local_index) = alx::io::locate_slice(requested_global_index, text_size, world_size);

        char last_row_character;
        if (target_rank == world_rank) {
          last_row_character = text_slice[local_index];
        } else {
          MPI_Get(&last_row_character, 1, MPI_CHAR, target_rank, local_index, 1, MPI_CHAR, window);
        }
        m_last_row[i] = last_row_character;
      }
    }
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    // Share primary index
    size_t shared_primary_index = 0;
    MPI_Allreduce(&m_primary_index, &shared_primary_index, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    m_primary_index = shared_primary_index;
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
  alx::ustring::value_type operator[](size_t i) const {
    return m_wm->operator[](i);
  }
  size_t primary_index() const {
    return m_primary_index;
  }
  size_t prev_occ(unsigned char c) const {
    return m_exclusive_prefix_histogram[c];
  }

  // Return size of bwt slice.
  size_t size() const {
    return m_end_index - m_start_index;
  }

  size_t global_rank(size_t global_pos, unsigned char c) const {
    size_t slice;
    size_t local_pos;
    std::tie(slice, local_pos) = alx::io::locate_bwt_slice(global_pos, m_global_size, m_world_size, m_primary_index);
    assert(slice == m_world_rank);
    // alx::io::alxout << "Answering rank. global_pos=" << global_pos << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
    return m_exclusive_prefix_histogram[c] + m_wm->rank(local_pos, c);
  }

  size_t next_border(size_t global_pos, unsigned char c) const {
    return m_first_row_starts[c] + global_rank(global_pos + 1, c);
  }

  void build_rank() {
    m_wm = std::make_unique<wm_type>(m_last_row.cbegin(), m_last_row.cend(), 256);
  }

  void free_bwt() {
    alx::ustring str;
    std::swap(m_last_row, str);
  }
  

};
}  // namespace alx
