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

class bwt_rle {
 private:
  size_t m_global_size;  // size of bwt
  size_t m_start_index;  // start of bwt slice
  size_t m_end_index;    // end of bwt slice (exclusive)
  int m_world_size;      // number of PEs
  int m_world_rank;      // PE number

  alx::ustring m_run_letters;                            // last row bwt matrix
  size_t m_primary_index;                                // index of implicit $ in last row
  std::array<size_t, 256> m_exclusive_prefix_histogram;  // histogram of text of previous PEs
  std::array<size_t, 256> m_first_row_starts;            // positions where the character runs start in global F

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(alx::ustring::const_iterator{}, alx::ustring::const_iterator{}, 256));
  std::unique_ptr<wm_type> m_run_letters_wm;  // wavelet tree to support rank

  // new in bwt_rle:
  std::vector<tdc::uint40_t> m_run_starts;                    // stored run start positions
  tdc::pred::Index<tdc::uint40_t> m_pred;                     // predecessor ds for run starts
  std::array<std::vector<tdc::uint40_t>, 256> m_run_lengths;  // stores prefix sum over length of all sigma-run, sorted by sigma
                                                              // std::array<tdc::uint40_t, 257> m_char_sum;                  equivalent to m_first_row_starts

 public:
  bwt_rle() : m_global_size{0}, m_start_index{0}, m_end_index{0} {}

  // Load partial bwt from bwt and primary index file.
  bwt_rle(std::filesystem::path const& last_row_path, std::filesystem::path const& primary_index_path) {
    bwt bwt(last_row_path, primary_index_path);
    build_struct(bwt);
  }

  // Calulate partial bwt from distributed suffix array and distributed text.

  template <typename t_text_container, typename t_sa_container>
  bwt_rle(t_text_container const& text_slice, t_sa_container const& sa_slice, size_t text_size, int world_rank, int world_size) {
    bwt bwt(sa_slice, text_size, world_rank, world_size);
    build_struct(bwt);
  }

  void build_struct(bwt const& input_bwt) {
    m_global_size = input_bwt.global_size();
    m_start_index = input_bwt.start_index();
    m_end_index = input_bwt.end_index();
    m_world_size = input_bwt.world_size();
    m_world_rank = input_bwt.world_rank();
    m_primary_index = input_bwt.primary_index();
    m_exclusive_prefix_histogram = input_bwt.exclusive_prefix_histogram();
    m_first_row_starts = input_bwt.first_row_starts();

    // Build m_run_letters, m_run_starts and m_run_lengths
    {
      // Prepare m_run_length
      for (size_t i = 0; i < 256; ++i) {
        m_run_lengths[i].push_back(0);
      }
      // Build data structure
      for (size_t run_start = 0; run_start < input_bwt.size();) {
        unsigned char run_letter = input_bwt.access_bwt(run_start);
        m_run_starts.push_back(run_start);
        m_run_letters.push_back(run_letter);

        size_t run_end = run_start + 1;
        while (run_end < input_bwt.size() && input_bwt.access_bwt(run_end) == run_letter) {
          ++run_end;
        }
        m_run_lengths[run_letter].push_back(m_run_lengths[run_letter].back() + run_end - run_start);
        run_start = run_end;
      }
      m_run_starts.shrink_to_fit();
      for (auto& a : m_run_lengths) {
        a.shrink_to_fit();
      }
    }
    // Build predecessor ds
    {
      m_pred = tdc::pred::Index<tdc::uint40_t>(m_run_starts.data(), m_run_starts.size(), 7);
    }
    // Build wavelet tree
    {
      m_run_letters_wm = std::make_unique<wm_type>(m_run_letters.begin(), m_run_letters.end(), 256);
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
  /*size_t last_row_size() {
    return m_run_letters.size();
  }
  alx::ustring::value_type access_bwt(size_t i) const {
    return m_run_letters[i];
  }
  alx::ustring::value_type access_wm(size_t i) const {
    return m_wm->operator[](i);
  }*/
  size_t primary_index() const {
    return m_primary_index;
  }

  // Return size of bwt slice.
  size_t size() const {
    return m_end_index - m_start_index;
  }

  size_t run_rank(unsigned char c, size_t i) const {
    // return m_run_letters_wt.rank(i, c);
    return m_run_letters_wm->rank(i, c); //i+1
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
    std::tie(slice, local_pos) = bwt::locate_bwt_slice(global_pos, m_global_size, m_world_size, m_primary_index);
    assert(slice == m_world_rank);
    // io::alxout << "Answering rank. global_pos=" << global_pos << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
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
      return std::get<0>(locate_bwt_slice(query.m_border.u64(), m_global_size, m_world_size, m_primary_index));
    }
  }

  static std::tuple<size_t, size_t> locate_bwt_slice(size_t global_index, size_t global_size, size_t world_size, size_t primary_index) {
    if (global_index > primary_index) {
      global_index--;
    }
    return alx::dist::locate_slice(global_index, global_size, world_size);
  }
};
}  // namespace alx::dist
