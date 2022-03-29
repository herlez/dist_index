#pragma once

#include <limits.h>
#include <mpi.h>
#include <stdint.h>

#include <array>
#include <bitset>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "timer.hpp"

namespace alx {
typedef std::basic_string<unsigned char> ustring;
}

namespace alx::dist {

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

int my_rank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int world_size() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return world_size;
}

// Calculate correct indexes [begin, end)
std::tuple<size_t, size_t> slice_indexes(size_t global_size, size_t world_rank, size_t world_size) {
  if (world_size > global_size) {
    if (world_rank < global_size) {
      return {world_rank, world_rank + 1};
    } else {
      return {global_size, global_size};
    }
  }

  size_t chunk_size = global_size / world_size;

  size_t begin = world_rank * chunk_size;
  size_t end = (world_rank != world_size - 1) ? (world_rank + 1) * chunk_size : global_size;

  return {begin, end};
}

// Calculate {node_rank, local_index}
std::tuple<size_t, size_t> locate_slice(size_t global_index, size_t global_size, size_t world_size) {
  if (world_size > global_size) {
    return {global_index, 0};
  }

  size_t chunk_size = global_size / world_size;

  size_t rank = global_index / chunk_size;
  rank = std::min(rank, world_size - 1);

  size_t local_index = (rank != world_size - 1) ? global_index % chunk_size : global_index - (world_size - 1) * chunk_size;

  // alxout << "global_index=" << global_index << " global_size=" << global_size << " world_size=" << world_size << " chunk_size=" << chunk_size << " rank=" << rank << " local_index=" << local_index << '\n';
  return {rank, local_index};
}
}  // namespace alx::dist

namespace alx::dist::io {
struct out_t {
  template <typename T>
  out_t& operator<<([[maybe_unused]] T&& x) {
    /*
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::filesystem::path log_path = "./alxlog" + std::to_string(world_rank) + std::string(".log");
    std::ofstream out(log_path, std::ios::out | std::ios::app);
    out << x;
    */
    return *this;
  }
};

static out_t alxout;

struct bench_out_t {
  template <typename T>
  bench_out_t& operator<<(T&& x) {
    if (my_rank() == 0) {
      std::cout << x;
    } else {
      std::filesystem::path log_path = "./alxbench" + std::to_string(my_rank()) + std::string(".log");
      std::ofstream out(log_path, std::ios::out | std::ios::app);
      out << x;
    }
    return *this;
  }
};

static bench_out_t benchout;

template <class T>
std::ostream& operator<<(std::ostream& o, const std::span<T>& span) {
  std::copy(span.begin(), span.end(), std::ostream_iterator<T>(o, " "));
  return o;
}

template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr) {
  std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "}";
  return out;
}

std::array<size_t, 256> histogram(std::string const& text) {
  std::array<size_t, 256> hist;
  hist.fill(0);
  for (unsigned char c : text) {
    ++hist[c];
  }
  return hist;
}

size_t alphabet_size(std::string const& text) {
  std::array<size_t, 256> hist = histogram(text);
  size_t alphabet_size = 0;
  for (size_t a : hist) {
    alphabet_size += (a > 0);
  }
  return alphabet_size;
}

size_t get_number_of_patterns(std::string header) {
  size_t start_pos = header.find("number=");
  if (start_pos == std::string::npos || start_pos + 7 >= header.size()) {
    return -1;
  }
  start_pos += 7;

  size_t end_pos = header.substr(start_pos).find(" ");
  if (end_pos == std::string::npos) {
    return -2;
  }

  return std::atoi(header.substr(start_pos).substr(0, end_pos).c_str());
}

size_t get_patterns_length(std::string header) {
  size_t start_pos = header.find("length=");
  if (start_pos == std::string::npos || start_pos + 7 >= header.size()) {
    return -1;
  }

  start_pos += 7;

  size_t end_pos = header.substr(start_pos).find(" ");
  if (end_pos == std::string::npos) {
    return -2;
  }

  size_t n = std::atoi(header.substr(start_pos).substr(0, end_pos).c_str());

  return n;
}

std::vector<std::string> load_patterns(std::filesystem::path path, size_t num_patterns = std::numeric_limits<size_t>::max()) {
  std::vector<std::string> patterns;
  std::ifstream ifs(path);

  std::string header;
  std::getline(ifs, header);

  size_t n = std::min(get_number_of_patterns(header), num_patterns);
  size_t m = get_patterns_length(header);

  // extract patterns from file and search them in the index
  size_t from = 0;
  size_t to = 0;  // n

  // std::tie(from, to) = slice_indexes(n, my_rank(), world_size());
  int innode_rank;
  MPI_Comm COMM_SHARED_MEMORY;
  MPI_Comm_split(MPI_COMM_WORLD, my_rank()/20, my_rank(), &COMM_SHARED_MEMORY);
  MPI_Comm_rank(COMM_SHARED_MEMORY, &innode_rank);
  
  int root_size, root_rank;
  int root_color = (innode_rank == 0) ? 0 : MPI_UNDEFINED;
  MPI_Comm COMM_ROOTS;
  MPI_Comm_split(MPI_COMM_WORLD, root_color, my_rank(), &COMM_ROOTS);
  if (innode_rank == 0) {
    MPI_Comm_size(COMM_ROOTS, &root_size);
    MPI_Comm_rank(COMM_ROOTS, &root_rank);
    std::tie(from, to) = slice_indexes(n, root_rank, root_size);
  }

  ifs.seekg(from * m, std::ios_base::cur);

  for (; from < to; ++from) {
    std::string p = std::string();

    for (size_t j = 0; j < m; ++j) {
      char c;
      ifs.get(c);
      p += c;
    }
    patterns.push_back(p);
  }

  return patterns;
}

}  // namespace alx::dist::io