#pragma once

#include <assert.h>
#include <mpi.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tdc/pred/index.hpp>
#include <tdc/uint/uint40.hpp>
#include <wavelet_tree/wavelet_tree.hpp>

#include "bwt.hpp"
#include "util/io.hpp"

namespace alx::dist {

class bwt_rle_eq {
 private:
  size_t m_global_size;  // size of bwt
  size_t m_start_index;  // start of bwt slice
  size_t m_end_index;    // end of bwt slice (exclusive)
  int m_world_size;      // number of PEs
  int m_world_rank;      // PE number

  alx::ustring m_run_letters;                            // last row bwt matrix
  std::array<size_t, 256> m_exclusive_prefix_histogram;  // histogram of text of previous PEs
  std::array<size_t, 256> m_first_row_starts;            // positions where the character runs start in global F

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(alx::ustring::const_iterator{}, alx::ustring::const_iterator{}, 256));
  std::unique_ptr<wm_type> m_run_letters_wm;  // wavelet tree to support rank

  // new in bwt_rle:
  std::vector<tdc::uint40_t> m_run_starts;                    // stored run start positions
  tdc::pred::Index<tdc::uint40_t> m_pred;                     // predecessor ds for run starts
  std::array<std::vector<tdc::uint40_t>, 256> m_run_lengths;  // stores prefix sum over length of all sigma-run, sorted by sigma
                                                              // std::array<tdc::uint40_t, 257> m_char_sum;                  equivalent to m_first_row_starts
  // new in bwt_rlq_eq
  std::vector<size_t> m_responsible;

 public:
  bwt_rle_eq() : m_global_size{0}, m_start_index{0}, m_end_index{0} {}

  // Load partial bwt from bwt file.
  bwt_rle_eq(std::filesystem::path const& text_path) {
    std::filesystem::path rlenc_path = text_path;
    rlenc_path.replace_extension(".rlenc");
    std::filesystem::path rlength_path = text_path;
    rlength_path.replace_extension(".rlength");
    // If file does not exist, return empty string.
    if (!std::filesystem::exists(rlenc_path)) {
      io::alxout << rlenc_path << " does not exist.";
      return;
    }
    if (!std::filesystem::exists(rlength_path)) {
      io::alxout << rlength_path << " does not exist.";
      return;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &m_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_world_rank);

    size_t rle_global_size = std::filesystem::file_size(rlenc_path);
    size_t rle_start_index, rle_end_index;
    std::tie(rle_start_index, rle_end_index) = slice_indexes(rle_global_size, m_world_rank, m_world_size);
    auto start_time = MPI_Wtime();
    {
      MPI_File handle_rlenc, handle_rlength;
      if (MPI_File_open(MPI_COMM_WORLD, rlenc_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &handle_rlenc) != MPI_SUCCESS) {
        std::cout << "Failure in opening the file.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      if (MPI_File_open(MPI_COMM_WORLD, rlength_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &handle_rlength) != MPI_SUCCESS) {
        std::cout << "Failure in opening the file.\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      m_run_letters.resize(rle_end_index - rle_start_index);
      m_run_starts.resize(rle_end_index - rle_start_index);
      // Because MPI_File_read_at_all only supports reading MAX_INT characters we do more iterations
      size_t read_already = 0;
      bool read_finished = (read_already == rle_end_index - rle_start_index);
      while (!read_finished) {
        size_t read_next = std::min(size_t{1} << 28, rle_end_index - rle_start_index - read_already);
        MPI_File_read_at_all(handle_rlenc, rle_start_index + read_already, m_run_letters.data() + read_already, read_next, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_read_at_all(handle_rlength, 5*(rle_start_index + read_already), m_run_starts.data() + read_already, read_next * 5, MPI_CHAR, MPI_STATUS_IGNORE);

        read_already += read_next;
        if (read_already == rle_end_index - rle_start_index) {
          read_finished = true;
        }
        MPI_Allreduce(MPI_IN_PLACE, &read_finished, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
      }
      MPI_File_close(&handle_rlenc);
      MPI_File_close(&handle_rlength);
    }
    auto end_time = MPI_Wtime();
    alx::dist::io::benchout << " bwt_load_time=" << static_cast<size_t>((end_time - start_time) * 1000);

    /*std::cout << "\n" << m_run_letters.size() << ": ";
    for(size_t i = 0; i < m_run_letters.size(); ++i) {
      std::cout << static_cast<uint64_t>(m_run_letters[i]) << " ";
    }
    std::cout << "\n" << m_run_starts.size() << ": ";
    for(size_t i = 0; i < m_run_starts.size(); ++i) {
      std::cout << m_run_starts[i].u64() << " ";
    }
    std::cout << "\n";*/

    // Build local histogram + run information. Use m_first_row_starts temporarily
    {
      m_exclusive_prefix_histogram.fill(0);
      m_first_row_starts.fill(0);
      for (size_t i = 0; i < 256; ++i) {
        m_run_lengths[i].push_back(0);
      }

      for (size_t i = 0; i < m_run_letters.size(); ++i) {
        unsigned char c = m_run_letters[i];
        size_t len = m_run_starts[i].u64();
        m_first_row_starts[c] += len;
        m_run_lengths[c].push_back(m_run_lengths[c].back() + len);
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
    size_t sum = 0;
    size_t old_sum = 0;
    for (size_t i = 0; i < m_run_starts.size(); ++i) {
      sum += m_run_starts[i].u64();
      m_run_starts[i] = old_sum;
      old_sum = sum;
    }

    // Build responsible array
    {
      size_t slice_size = std::accumulate(m_first_row_starts.begin(), m_first_row_starts.end(), 0);
      m_responsible.resize(m_world_size + 1);
      m_responsible[m_world_rank] = slice_size;
      MPI_Allreduce(MPI_IN_PLACE, m_responsible.data(), m_world_size, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
      std::exclusive_scan(m_responsible.begin(), m_responsible.end(), m_responsible.begin(), 0);
      m_global_size = m_responsible.back();
      m_start_index = m_responsible[m_world_rank];
      m_end_index = m_responsible[m_world_rank + 1];
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

    // Build predecessor ds
    {
      m_pred = tdc::pred::Index<tdc::uint40_t>(m_run_starts.data(), m_run_starts.size(), 7);
    }
    // Build wavelet tree
    {
      m_run_letters_wm = std::make_unique<wm_type>(m_run_letters.begin(), m_run_letters.end(), 256);
    }
    io::alxout << m_global_size << "\n";
    io::alxout << m_start_index << "\n";
    io::alxout << m_end_index << "\n";
    io::alxout << m_world_size << "\n";
    io::alxout << m_world_rank << "\n";

    for (size_t i = 0; i < m_run_letters.size(); ++i) {
      io::alxout << static_cast<uint64_t>(m_run_letters[i]) << " ";
    }
    io::alxout << "\n";
    for (size_t i = 0; i < m_exclusive_prefix_histogram.size(); ++i) {
      io::alxout << m_world_rank << ": " << m_exclusive_prefix_histogram[i] << " ";
    }
    io::alxout << "\n";
    for (size_t i = 0; i < m_first_row_starts.size(); ++i) {
      io::alxout << m_world_rank << ": " << m_first_row_starts[i] << " ";
    }
    io::alxout << "\n";

    for (size_t i = 0; i < m_run_starts.size(); ++i) {
      io::alxout << m_run_starts[i].u64() << " ";
    }
    io::alxout << "\n";
    for (auto const& a : m_run_lengths) {
      if (a.size() > 1) {
        for (size_t i = 0; i < a.size(); ++i) {
          io::alxout << a[i] << " ";
        }
        io::alxout << "\n";
      }
    }

    io::alxout << "\n";
    for (size_t i = 0; i < m_responsible.size(); ++i) {
      io::alxout << m_responsible[i] << " ";
    }
    io::alxout << "\n";
  }

  // Getter
  size_t
  global_size() const {
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

  // Return size of bwt slice.
  size_t size() const {
    return m_end_index - m_start_index;
  }

  size_t run_rank(unsigned char c, size_t i) const {
    // return m_run_letters_wt.rank(i, c);
    return m_run_letters_wm->rank(i, c);  // i+1
  }

  size_t pred(size_t i) const {
    return m_pred.predecessor(m_run_starts.data(), m_run_starts.size(), i).pos;
  }

  size_t local_rank(size_t local_pos, unsigned char c) const {
    size_t kth_run = pred(local_pos);
    size_t run_start = m_run_starts[kth_run].u64();
    size_t num_c_run = run_rank(c, kth_run);
    unsigned char run_symbol = m_run_letters[kth_run];
    return size_t{m_run_lengths[c][num_c_run]} + (c == run_symbol ? (local_pos - run_start) : 0);
    // return m_run_letters_wm->rank(local_pos, c);
  }

  size_t global_rank(size_t global_pos, unsigned char c) const {
    size_t slice;
    size_t local_pos;
    std::tie(slice, local_pos) = locate_bwt_slice(global_pos, m_global_size, m_world_size);
    assert(slice == m_world_rank);
    io::alxout << "Answering rank. global_pos=" << global_pos << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
    // std::cout << "Answering rank. global_pos=" << global_pos << " slice=" << slice << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
    return m_exclusive_prefix_histogram[c] + local_rank(local_pos, c);
  }

  size_t next_border(size_t global_pos, unsigned char c) const {
    return m_first_row_starts[c] + global_rank(global_pos, c);
  }

  // Is needed anyway
  void build_rank() {
  }

  // Not that useful since last_row_rle only takes up r*n bytes.
  void free_bwt() {
  }

  template <typename t_query>
  int get_target_pe(t_query const& query) const {
    if (query.m_pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(locate_bwt_slice(query.m_border.u64(), m_global_size, m_world_size));
    }
  }

  std::tuple<size_t, size_t> locate_bwt_slice(size_t global_index, [[maybe_unused]] size_t global_size, [[maybe_unused]] size_t world_size) const {
    int target_pe = std::distance(m_responsible.begin(), std::upper_bound(m_responsible.begin(), m_responsible.end(), global_index)) - 1;
    target_pe = std::min(target_pe, (int)world_size - 1);
    return {target_pe, global_index - m_responsible[target_pe]};
    // return alx::dist::locate_slice(global_index, global_size, world_size);
  }
};
}  // namespace alx::dist
