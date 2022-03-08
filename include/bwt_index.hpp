#pragma once

#include <assert.h>
#include <mpi.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <wavelet_tree/wavelet_tree.hpp>

#include "bwt.hpp"
#include "query.hpp"
#include "query_information.hpp"
#include "util/io.hpp"

namespace alx::dist {

/*
class concatenated_strings {
 public:
  concatenated_strings() = default;

  concatenated_strings(std::vector<std::string> strings) {
    num_strings = strings.size();
    starts_resize(num_strings);
    if(num_strings != 0) {
      len_strings = num_strings * strings[0].size();
      concat_string_resize(len_strings);
    }

    size_t last_str_pos = 0;
    for(size_t i = 0; i < num_strings; ++i) {
      auto const& str = strings[i];
      starts[i] = last_str_pos;
      std::copy(str.begin(), str.end(), concat_string_data() + last_str_pos);

      last_str_pos += str.size();
    }
  }

  size_t size() const {
    return num_strings;
  }

  size_t length_all_strings() const {
    return len_strings;
  }

  char* concat_string_data() {
    return concat_string.data();
  }

  void concat_string_resize(size_t k) {
    len_strings = k;
    concat_string.resize(k);
  }
  uint32_t* starts_data() {
    return starts.data();
  }
  void starts_resize(size_t k) {
    num_strings = k;
    starts.resize(k);
  }
  std::vector<char>::const_iterator operator[](size_t i) const {
    return concat_string.begin() + starts[i];
  }
  size_t num_strings;
  size_t len_strings;
  std::vector<char> concat_string;
  std::vector<uint32_t> starts;
};*/


class concatenated_strings {
 public:
  concatenated_strings() = default;

  concatenated_strings(std::vector<std::string> strings) {
    starts_resize(strings.size());
    concat_string_resize(strings.size() * strings[0].size());
    
    size_t last_str_pos = 0;
    for(size_t i = 0; i < num_strings; ++i) {
      auto const& str = strings[i];
      starts[i] = last_str_pos;
      std::copy(str.begin(), str.end(), concat_string_data() + last_str_pos);

      last_str_pos += str.size();
    }
    io::alxout << "Concat_string built. len=" << len_strings << "\n";
  }

  ~concatenated_strings() {
    delete[] concat_string;
    delete[] starts;
  }

  size_t size() const {
    return num_strings;
  }

  size_t length_all_strings() const {
    return len_strings;
  }

  char* concat_string_data() {
    return concat_string;
  }

  void concat_string_resize(size_t k) {
    if(len_strings < k) {
      len_strings = k;
      concat_string = new char[k];
      io::alxout << "Concat_string resized to " << len_strings << "\n";
    }
  }

  uint32_t* starts_data() {
    return starts;
  }
  void starts_resize(size_t k) {
      if(num_strings < k) {
      num_strings = k;
      starts = new uint32_t[k];
      io::alxout << "Start resized to " << num_strings << "\n";
    }
  }

  char* operator[](size_t i) const {
    return (concat_string + starts[i]);
  }

  size_t num_strings = 0;
  size_t len_strings = 0;
  char* concat_string = nullptr;
  uint32_t* starts = nullptr;
};

template <typename t_bwt>
class bwt_index {
 private:
  t_bwt const* m_bwt;

  // std::unordered_map<__int128_t, std::pair<size_t, size_t>> m_headstart;
  bool m_headstart_avail = false;
  constexpr static size_t m_static_headstart = 1;
  // std::array<std::pair<size_t, size_t>, (size_t{1} << (m_static_headstart * 8))> m_headstart;
  std::unordered_map<size_t, std::pair<size_t, size_t>> m_headstart;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt file.
  bwt_index(t_bwt const& bwt) : m_bwt(&bwt) {
    static_headstart();
    m_headstart_avail = true;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void dynamic_headstart() {
    //...
  }

  void static_headstart() {
    std::vector<std::string> patterns;

    if (my_rank() == 0) {
      patterns.reserve(m_headstart.size());
      std::string next_string;
      next_string.resize(m_static_headstart);
      // for (size_t i = 0; i < 256; ++i) {
      // for (size_t j = 0; j < 256; ++j) {
      for (size_t k = 0; k < 256; ++k) {
        next_string[0] = (char)k;
        // next_string[1] = (char)j;
        // next_string[2] = (char)i;
        patterns.push_back(next_string);
      }
      //}
      //}
    }

    std::vector<rank_query> finished_queries = occ<rank_query>(patterns);

    if (my_rank() == 0) {
      assert(finished_queries.size() == 2 * patterns.size());
      std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
        return left.m_id < right.m_id;
      });

      for (size_t i = 0; i < finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[i].m_border.u64(), finished_queries[i + 1].m_border.u64());
        m_headstart[i / 2].first = left;
        m_headstart[i / 2].second = right;

        io::alxout << "Result: " << right - left << "\n";
      }
    }
  }

  std::vector<size_t>
  occ_batched_preshared(std::vector<std::string> const& patterns) {
    // Build concatenated string and share between PEs
    concatenated_strings conc_strings(patterns);
    io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
    io::alxout << conc_strings.concat_string << '\n';
    io::alxout << conc_strings.starts << '\n';

    // Share conc_string.concat_string
    {
      size_t conc_strings_size = conc_strings.length_all_strings();
      MPI_Bcast(&conc_strings_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
      conc_strings.concat_string_resize(conc_strings_size);
      MPI_Bcast(conc_strings.concat_string_data(), conc_strings_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    // Broadcast conc_string.starts
    {
      size_t num_patterns = patterns.size();
      MPI_Bcast(&num_patterns, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
      conc_strings.starts_resize(num_patterns);
      MPI_Bcast(conc_strings.starts_data(), num_patterns, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }
    io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
    io::alxout << conc_strings.concat_string << '\n';
    io::alxout << conc_strings.starts << '\n';

    // Prepare results
    std::vector<size_t> results;
    results.reserve(patterns.size());

    std::vector<rank_query_information> finished_queries = occ<rank_query_information>(patterns, conc_strings);

    if (my_rank() == 0) {
      assert(finished_queries.size() == 2 * patterns.size());
      std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
        return left.m_id < right.m_id;
      });

      for (size_t i = 0; i < finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[i].m_border.u64(), finished_queries[i + 1].m_border.u64());
        io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    results.reserve(patterns.size());

    std::vector<rank_query> finished_queries = occ<rank_query>(patterns);

    if (my_rank() == 0) {
      assert(finished_queries.size() == 2 * patterns.size());
      std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
        return left.m_id < right.m_id;
      });

      for (size_t i = 0; i < finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[i].m_border.u64(), finished_queries[i + 1].m_border.u64());
        io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  std::vector<size_t> occ_one_by_one(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    results.reserve(patterns.size());

    // Answer single queries
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    io::alxout << "#Patterns=: " << patterns_size << "\n";

    for (size_t i = 0; i < patterns_size; ++i) {
      size_t num_queries = (my_rank() == 0) ? 1 : 0;
      std::span single_pattern(patterns.begin() + i, num_queries);
      io::alxout << "Queries: " << single_pattern << "\n";

      std::vector<rank_query> finished_queries = occ<rank_query>(single_pattern);

      if (my_rank() == 0) {
        assert(finished_queries[0].m_pos_in_pattern == 0 && finished_queries[1].m_pos_in_pattern == 0);
        assert(finished_queries.size() == 2 * single_pattern.size());
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[0].m_border.u64(), finished_queries[1].m_border.u64());
        io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  template <typename t_query>
  std::vector<t_query> occ(auto const& patterns, [[maybe_unused]] concatenated_strings const& conc_strings = concatenated_strings()) {
    MPI_Datatype query_mpi_type = t_query::mpi_type();
    MPI_Type_commit(&query_mpi_type);

    std::vector<t_query> queries(initialize_queries<t_query>(patterns));
    std::vector<t_query> queries_buffer;
    std::vector<t_query> finished_queries;
    finished_queries.reserve(patterns.size());

    io::alxout << '\n'
               << queries << '\n';

    // Loop : Send and Calculate until finished
    bool global_finished = false;
    while (!global_finished) {
      // Send queries with Alltoallv
      {
        // Sort queries by (are they finished?, border)
        std::sort(queries.begin(), queries.end(), [](auto const& left, auto const& right) {
          if (left.m_pos_in_pattern == 0 && right.m_pos_in_pattern != 0) return true;
          if (left.m_pos_in_pattern != 0 && right.m_pos_in_pattern == 0) return false;
          return (left.m_border < right.m_border);
        });

        // Define my counts for sending (how many integers do I send to each process?)
        std::vector<int> counts_send(world_size());
        for (size_t i = 0; i < queries.size(); ++i) {
          int target_pe = m_bwt->get_target_pe(queries[i]);
          ++counts_send[target_pe];
        }
        io::alxout << "#SEND: " << counts_send << '\n';

        // Define my counts for receiving (how many integers do I receive from each process?)
        std::vector<int> counts_recv(world_size());
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Alltoall(counts_send.data(), 1, MPI_INT, counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
        io::alxout << "#RECV: " << counts_recv << '\n';

        // Define my displacements for sending (where is located in the buffer each message to send?)
        std::vector<int> displacements_send(world_size());
        std::exclusive_scan(counts_send.begin(), counts_send.end(), displacements_send.begin(), 0);
        io::alxout << "DSEND: " << displacements_send << '\n';

        // Define my displacements for reception (where to store in buffer each message received?)
        std::vector<int> displacements_recv(world_size());
        std::exclusive_scan(counts_recv.begin(), counts_recv.end(), displacements_recv.begin(), 0);
        io::alxout << "DRECV: " << displacements_recv << '\n';

        int num_incoming_queries = displacements_recv.back() + counts_recv.back();
        queries_buffer.resize(num_incoming_queries);

        MPI_Alltoallv(queries.data(), counts_send.data(), displacements_send.data(), query_mpi_type, queries_buffer.data(), counts_recv.data(), displacements_recv.data(), query_mpi_type, MPI_COMM_WORLD);
        std::swap(queries, queries_buffer);
        io::alxout << queries << '\n';
      }

      // Calculate
      {
        for (size_t i = 0; i < queries.size(); ++i) {
          auto& query = queries[i];

          while ((m_bwt->get_target_pe(query) == my_rank()) && query.m_pos_in_pattern != 0) {
            query.m_pos_in_pattern--;

            if constexpr (std::is_same<t_query, rank_query_information>::value) {
              char cur_char = conc_strings[query.m_id][query.m_pos_in_pattern];
              query.m_border = m_bwt->next_border(query.m_border.u64(), cur_char);
            } else if constexpr (std::is_same<t_query, rank_query>::value) {
              char cur_char = query.cur_char();
              query.m_border = m_bwt->next_border(query.m_border.u64(), cur_char);
            } else {
              std::cerr << "Unknown query type!\n";
            }

            io::alxout << "Calculated query[" << i << "]: " << query << "\n";
          }
          // Move answered queries
          if ((my_rank() == 0) && query.m_pos_in_pattern == 0) {
            finished_queries.push_back(query);
            query = queries.back();
            queries.pop_back();
            --i;
          }
        }
      }

      bool local_finished = queries.empty();
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }

    MPI_Type_free(&query_mpi_type);
    return finished_queries;
  }

  template <typename t_query>
  std::vector<t_query> initialize_queries(auto const& patterns) {
    std::vector<t_query> queries;
    queries.reserve(patterns.size() * 2);
    io::alxout << "Initializing " << patterns.size() << " queries..\n";

    if (m_headstart_avail) {
      for (size_t i = 0; i < patterns.size(); ++i) {
        size_t pattern_suffix = 0;
        size_t suffix_length = 0;
        for (auto rev_iterator = patterns[i].rbegin(); rev_iterator != patterns[i].rbegin() + m_static_headstart; ++rev_iterator) {
          pattern_suffix *= 256;
          pattern_suffix += static_cast<unsigned char>(*rev_iterator);
          ++suffix_length;
        }

        while (m_headstart.find(pattern_suffix) == m_headstart.end()) {
          pattern_suffix >>= 8;
          suffix_length--;
        }

        queries.emplace_back(patterns[i], patterns[i].size() - suffix_length, m_headstart[pattern_suffix].first, i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
        queries.emplace_back(patterns[i], patterns[i].size() - suffix_length, m_headstart[pattern_suffix].second, i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      }
    } else {
      for (size_t i = 0; i < patterns.size(); ++i) {
        queries.emplace_back(patterns[i], patterns[i].size(), 0, i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
        queries.emplace_back(patterns[i], patterns[i].size(), m_bwt->global_size(), i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      }
    }
    return queries;
  }
};

}  // namespace alx::dist