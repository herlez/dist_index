#pragma once

#include <assert.h>
#include <libsais64.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "util/io.hpp"

namespace alx {

struct bwt {
  size_t m_start_index;
  size_t m_end_index;
  size_t m_chunk_size;

  size_t primary_index;
  std::string last_row;

  bwt() = default;

  bwt(std::string const& text) {
    last_row.resize(text.size());
    std::vector<int64_t> sa(text.size() + 100);
    primary_index = libsais64_bwt(reinterpret_cast<const uint8_t*>(text.data()), reinterpret_cast<uint8_t*>(last_row.data()), sa.data(), text.size(), int64_t{100}, nullptr);
  }

  std::string to_text() const {
    std::string text(last_row.size(), ' ');
    std::vector<int64_t> sa(text.size() + 1);
    libsais64_unbwt(reinterpret_cast<const uint8_t*>(last_row.data()), reinterpret_cast<uint8_t*>(text.data()), sa.data(), text.size(), nullptr, primary_index);
    return text;
  }

  int to_file(std::filesystem::path last_row_path, std::filesystem::path primary_index_path) const {
    alx::io::alxout << "Write to " << last_row_path << " and " << primary_index_path << "\n";
    // Write bwt
    if (!std::filesystem::exists(primary_index_path)) {
      std::ofstream out(primary_index_path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(&primary_index), sizeof(primary_index));
    }
    if (!std::filesystem::exists(last_row_path)) {
      std::ofstream out(last_row_path, std::ios::binary);
      out << last_row;
    }
    return 0;
  }

  bwt(std::filesystem::path const& last_row_path, std::filesystem::path const& primary_index_path, int world_rank, int world_size) {
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
      in.read(reinterpret_cast<char*>(&primary_index), sizeof(primary_index));
    }
    // Read last row
    {
      std::ifstream in(last_row_path, std::ios::binary);
      in.seekg(0, std::ios::beg);
      std::streampos begin = in.tellg();
      in.seekg(0, std::ios::end);
      size_t size = in.tellg() - begin;
      m_chunk_size = size / world_size;
      m_start_index = world_rank * m_chunk_size;
      m_end_index = (world_rank == world_size - 1) ? size : (world_rank + 1) * m_chunk_size;

      last_row.resize(size);
      in.seekg(m_start_index, std::ios::beg);
      in.read(last_row.data(), m_end_index - m_start_index);
    }
  }

  template <typename SA_Container>
  static bwt bwt_from_sa(SA_Container const& sa, std::string const& text) {
    assert(sa.size() == text.size());
    alx::bwt bwt;
    bwt.last_row.reserve(text.size());
    bwt.last_row.push_back(text.back());  // In first_row $ is first.

    for (size_t i{0}; i < sa.size(); ++i) {
      if (sa[i] == 0) [[unlikely]] {
        bwt.primary_index = i;
      } else {
        bwt.last_row.push_back(text[sa[i] - 1]);
      }
    }
    assert(bwt.size() == text.size());
    return bwt;
  }

  size_t size() const {
    return last_row.size();
  }

  /*char& operator[](size_t i) {
    return last_row[i];
  }*/
};
}  // namespace alx
