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

class bwt {
 private:
  size_t m_start_index;
  size_t m_end_index;
  size_t m_global_size;

  size_t m_primary_index;
  alx::ustring m_last_row;
  std::array<size_t, 256> m_prev_occ;

 public:
  // Load partial bwt from bwt and primary index file.
  bwt(std::filesystem::path const& last_row_path, std::filesystem::path const& primary_index_path, int world_rank, int world_size) {
    // If file does not exist, return empty string
    if (!std::filesystem::exists(last_row_path)) {
      alx::io::alxout << last_row_path << " does not exist.";
      return;
    }
    if (!std::filesystem::exists(primary_index_path)) {
      alx::io::alxout << primary_index_path << " does not exist.";
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

  // Calulate partial bwt from distributed suffix array and distributed text.
  template <typename t_text_container, typename t_sa_container>
  bwt(t_text_container const& text_slice, t_sa_container const& sa_slice, size_t text_size, int world_rank, int world_size) {
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

  size_t prev_occ(unsigned char c) const {
    return m_prev_occ[c];
  }

  size_t primary_index() const {
    return m_primary_index;
  }

  size_t start_index() const {
    return m_start_index;
  }

  size_t end_index() const {
    return m_end_index;
  }

  // Return size of text/bwt.
  size_t global_size() const {
    return m_last_row.size();
  }

  // Return size of text/bwt slice.
  size_t size() const {
    return m_end_index - m_start_index;
  }

  alx::ustring::const_iterator cbegin() const {
    return m_last_row.cbegin();
  }

  alx::ustring::const_iterator cend() const {
    return m_last_row.cbegin();
  }

  alx::ustring::value_type const& operator[](size_t i) const {
    return m_last_row[i];
  }
};
}  // namespace alx
