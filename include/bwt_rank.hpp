#pragma once

#include "bwt.hpp"

namespace alx {

class bwt_rank {
 private:
  size_t m_global_size;  // size of bwt
  size_t m_start_index;  // start of bwt slice
  size_t m_end_index;    // end of bwt slice (exclusive)
  int m_world_size;      // number of PEs
  int m_world_rank;      // PE number

  size_t m_primary_index;                                // index of implicit $ in last row
  std::array<size_t, 256> m_exclusive_prefix_histogram;  // histogram of text of previous PEs
  std::array<size_t, 256> m_first_row_starts;            // positions where the character runs start in F

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(alx::ustring::const_iterator{}, alx::ustring::const_iterator{}, 256));
  std::unique_ptr<wm_type> m_wm;  // wavelet tree to support rank

 public:
  bwt_rank() : m_global_size{0}, m_start_index{0}, m_end_index{0} {}

  // Load partial bwt from bwt and primary index file.
  bwt(alx::bwt const& bwt) {
    m_global_size = bwt.global_size();
    m_start_index = bwt.start_index();
    m_end_index = bwt.end_index();
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    m_primary_index = bwt.primary_index;


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
    // alx::io::alxout << "Answeing rank. global_pos=" << global_pos << " local_pos=" << local_pos << " world_size=" << m_world_size << " c=" << c << "\n";
    return m_exclusive_prefix_histogram[c] + m_wm->rank(local_pos, c);
  }

  size_t next_border(size_t global_pos, unsigned char c) const {
    return m_first_row_starts[c] + global_rank(global_pos + 1, c);
  }

 private:
  void build_rank(alx::ustring const& last_row) {
    m_wm = std::make_unique<wm_type>(last_row.cbegin(), last_row.cend(), 256);
  }
};
}  // namespace alx
