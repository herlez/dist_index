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
#include "util/string_enumerator.hpp"

namespace alx::dist {

class concatenated_strings {
 public:
  concatenated_strings() = default;

  concatenated_strings(std::vector<std::string> strings) {
    // strings_num
    const size_t local_strings_num = strings.size();
    MPI_Allreduce(&local_strings_num, &strings_num, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    // strings_len
    size_t local_strings_len = 0;
    for (auto const& str : strings) {
      local_strings_len += str.size();
    }
    MPI_Allreduce(&local_strings_len, &strings_len, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    io::alxout << "local_strings_num=" << local_strings_num << " local_strings_len=" << local_strings_len << "\n";

    // Prepare shared memory windows
    int innode_size, innode_rank;
    MPI_Comm COMM_SHARED_MEMORY;
    MPI_Comm_split(MPI_COMM_WORLD, my_rank() / 20, my_rank(), &COMM_SHARED_MEMORY);
    MPI_Comm_size(COMM_SHARED_MEMORY, &innode_size);
    MPI_Comm_rank(COMM_SHARED_MEMORY, &innode_rank);
    // std::cout << my_rank() << " innode_size=" << innode_size << " innode_rank=" << innode_rank << "\n";

    int root_size, root_rank;
    int root_color = (innode_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm COMM_ROOTS;
    MPI_Comm_split(MPI_COMM_WORLD, root_color, my_rank(), &COMM_ROOTS);
    if (innode_rank == 0) {
      MPI_Comm_size(COMM_ROOTS, &root_size);
      MPI_Comm_rank(COMM_ROOTS, &root_rank);
      // std::cout << my_rank() << " root_size=" << root_size << " root_rank=" << root_rank << "\n";
    }

    // Open shared memory windows for starts and conc_string
    {
      const int starts_size = (innode_rank == 0) ? strings_num : 0;
      const int conc_size = (innode_rank == 0) ? strings_len : 0;
      MPI_Win_allocate_shared(starts_size * sizeof(uint32_t), sizeof(uint32_t), MPI_INFO_NULL, COMM_SHARED_MEMORY, &starts, &window_st);
      MPI_Win_allocate_shared(conc_size, sizeof(char), MPI_INFO_NULL, COMM_SHARED_MEMORY, &concat_string, &window_cc);
      MPI_Aint full_win_size;
      int full_win_disp;
      MPI_Win_shared_query(window_st, 0, &full_win_size, &full_win_disp, &starts);
      io::alxout << my_rank() << " win_size=" << full_win_size << " win_disp=" << full_win_disp << "\n";
      MPI_Win_shared_query(window_cc, 0, &full_win_size, &full_win_disp, &concat_string);
      io::alxout << my_rank() << " win_size=" << full_win_size << " win_disp=" << full_win_disp << "\n";

      // Initialize share memory window
      if (innode_rank == 0) {
        for (size_t i = 0; i < strings_num; ++i) {
          starts[i] = 0;
        }
        for (size_t i = 0; i < strings_len; ++i) {
          concat_string[i] = 0;
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    print();

    // Local buffers
    std::vector<uint32_t> local_starts;
    local_starts.resize(local_strings_num);
    std::vector<char> local_concat;
    local_concat.resize(local_strings_len);

    // Prepare
    size_t strings_before = 0;
    size_t chars_before = 0;
    MPI_Exscan(&local_strings_num, &strings_before, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&local_strings_len, &chars_before, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    io::alxout << "strings_before=" << strings_before << " chars_before=" << chars_before << "\n";

    // Fill local buffers
    size_t chars_before_it = chars_before;
    for (size_t i = 0; i < strings.size(); ++i) {
      local_starts[i] = chars_before_it;
      chars_before_it += strings[i].size();

      std::copy(strings[i].begin(), strings[i].end(), local_concat.data() + local_starts[i] - chars_before);
    }
    io::alxout << "local_starts: " << local_starts << "\nlocal_concat: " << local_concat << "\n";

    // Share starts with shared memory PEs
    std::vector<int> recvcounts;
    recvcounts.resize(innode_size);
    int sendcount = local_strings_num;
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, COMM_SHARED_MEMORY);

    std::vector<int> displs;
    displs.resize(innode_size);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), displs.begin(), strings_before);
    MPI_Gatherv(local_starts.data(), local_strings_num, MPI_UINT32_T, starts, recvcounts.data(), displs.data(), MPI_UINT32_T, 0, COMM_SHARED_MEMORY);
    print();

    // Share starts with other shared memory roots
    if (innode_rank == 0) {
      MPI_Allreduce(MPI_IN_PLACE, starts, strings_num, MPI_UINT32_T, MPI_SUM, COMM_ROOTS);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    print();

    // Share conc with shared memory PEs
    sendcount = local_strings_len;
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, COMM_SHARED_MEMORY);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), displs.begin(), chars_before);
    MPI_Gatherv(local_concat.data(), local_strings_len, MPI_CHAR, concat_string, recvcounts.data(), displs.data(), MPI_CHAR, 0, COMM_SHARED_MEMORY);

    // Share conc with other shared memory roots
    if (innode_rank == 0) {
      MPI_Allreduce(MPI_IN_PLACE, concat_string, strings_len, MPI_CHAR, MPI_SUM, COMM_ROOTS);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    print();
    MPI_Comm_free(&COMM_SHARED_MEMORY);
    if (innode_rank == 0) {
      MPI_Comm_free(&COMM_ROOTS);
    }
  }

  void destroy() {
    MPI_Win_free(&window_cc);
    MPI_Win_free(&window_st);
  }

  void print() {
    for (size_t i = 0; i < strings_num; ++i) {
      io::alxout << starts[i] << " ";
    }
    io::alxout << "\n";
    for (size_t i = 0; i < strings_len; ++i) {
      io::alxout << concat_string[i];
    }
    io::alxout << "\n";
  }

  size_t size() const {
    return strings_num;
  }

  size_t length_all_strings() const {
    return strings_len;
  }

  char* concat_string_data() {
    return concat_string;
  }

  uint32_t* starts_data() {
    return starts;
  }

  char* operator[](size_t i) const {
    return (concat_string + starts[i]);
  }

  size_t strings_num = 0;
  size_t strings_len = 0;
  char* concat_string = nullptr;
  uint32_t* starts = nullptr;
  MPI_Win window_cc;
  MPI_Win window_st;
};

std::string code_to_string(size_t code) {
  std::string str;
  for (size_t i = 0; i < 8; ++i) {
    size_t letter = (code >> ((7 - i) * 8)) & 255;
    if (letter != 0) {
      str.push_back(letter);
    }
  }
  return str;
}

template <typename t_bwt>
class bwt_index {
 private:
  t_bwt const* m_bwt;

  bool m_head_start_avail = false;
  bool m_head_start_dynamic = false;
  size_t max_head_start_entries;
  std::unordered_map<size_t, std::pair<size_t, size_t>> m_head_start;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt file.
  bwt_index(t_bwt const& bwt, bool head_start_dynamic = false, size_t head_start_size = 1'000)
      : m_bwt(&bwt), m_head_start_dynamic(head_start_dynamic), max_head_start_entries(head_start_size) {
    auto start_time = MPI_Wtime();

    m_head_start[0] = {0, m_bwt->global_size()};
    if(head_start_dynamic) {
      dynamic_head_start();
    } else {
      static_head_start();
    }
    m_head_start_avail = true;
    auto end_time = MPI_Wtime();
    io::benchout << " head_start_dynamic=" << m_head_start_dynamic << " head_start_size=" << m_head_start.size() << " headstart_time=" << static_cast<size_t>((end_time - start_time) * 1000);

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void dynamic_head_start() {
    //...
  }

  void static_head_start() {
    std::vector<std::string> patterns;

    if (my_rank() == 0) {
      // Calculate alphabet
      std::array<size_t, 256> first_row_starts = m_bwt->first_row_starts();
      std::vector<unsigned char> alphabet;
      for (size_t i = 1; i < first_row_starts.size(); ++i) {
        if (first_row_starts[i] > first_row_starts[i - 1]) {
          alphabet.push_back(i - 1);
        }
      }
      if (m_bwt->size() > first_row_starts.back()) {
        alphabet.push_back(255);
      }
      io::benchout << " sigma=" << alphabet.size();

      // Enumerate over all strings form this alphabet up to max_head_start_entries strings or t_headstart depth
      util::string_enumerator str_enumerator(alphabet);
      for (size_t i = 0; i < max_head_start_entries; ++i) {
        patterns.push_back(str_enumerator.get());
        str_enumerator.next();
      }
      io::benchout << " max_headstart_depth=" << size_t{str_enumerator.current_length()};
    }
    /*for (auto const& p : patterns) {
      std::cout << p << "\n";
    }*/

    // SHARE FINISHED QUERIES AND BUILD LOOKUP MAP TODO
    m_head_start[0] = {0, m_bwt->global_size()};

    std::vector<rank_query> finished_queries = occ<rank_query>(patterns);

    if (my_rank() == 0) {
      assert(finished_queries.size() == 2 * patterns.size());
      std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
        return left.m_id < right.m_id;
      });
      /*for (auto & q : finished_queries) {
        std::cout << q << ": " << q.get_code() << "\n";
      }*/

      for (size_t i = 0; i < finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[i].m_border.u64(), finished_queries[i + 1].m_border.u64());

        size_t code = finished_queries[i].get_code();
        m_head_start[code].first = left;
        m_head_start[code].second = right;

        io::alxout << "Result: " << right - left << "\n";
      }
    }
  }

  std::vector<size_t>
  occ_batched_preshared(std::vector<std::string> const& patterns) {
    auto start_time = MPI_Wtime();
    concatenated_strings conc_strings(patterns);
    auto end_time = MPI_Wtime();
    io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
    io::benchout << " preshare_time=" << static_cast<size_t>((end_time - start_time) * 1000);

    // Prepare results
    std::vector<size_t> results;
    results.reserve(patterns.size());

    std::vector<rank_query_information> finished_queries = occ<rank_query_information>(patterns, conc_strings);
    conc_strings.destroy();

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

    MPI_Barrier(MPI_COMM_WORLD);
    return results;
  }

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    size_t local_patterns_size = patterns.size();
    size_t global_patterns_size = 0;
    MPI_Allreduce(&local_patterns_size, &global_patterns_size, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    std::vector<size_t> results;
    if (my_rank() == 0) {
      results.reserve(global_patterns_size);
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
    if (my_rank() == 0) {
      finished_queries.reserve(patterns.size());
    }

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
    size_t start_id = 0;
    size_t local_patterns_size = patterns.size();
    MPI_Exscan(&local_patterns_size, &start_id, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    std::vector<t_query> queries;
    queries.reserve(patterns.size() * 2);
    io::alxout << "Initializing " << patterns.size() << " queries..\n";

    if (m_head_start_avail) {
      for (size_t i = 0; i < patterns.size(); ++i) {
        auto const& pat = patterns[i];
        size_t pattern_suffix = 0;
        size_t suffix_length = 0;

        while (m_head_start.contains(pattern_suffix) && suffix_length < 8) {
          ++suffix_length;
          pattern_suffix += (pat[pat.size() - suffix_length] << ((suffix_length - 1) * 8));
          io::alxout << "Lets try: " << code_to_string(pattern_suffix) << "\n";
        }
        if (!m_head_start.contains(pattern_suffix)) {
          pattern_suffix -= (pat[pat.size() - suffix_length] << ((suffix_length - 1) * 8));
          --suffix_length;
          io::alxout << "It failed, so we take: " << code_to_string(pattern_suffix) << "\n";
        }

        /*
        size_t pattern_suffix = 0;
        size_t suffix_length = 0;
        for (auto rev_iterator = patterns[i].rbegin(); rev_iterator != patterns[i].rbegin() + m_static_headstart; ++rev_iterator) {
          pattern_suffix *= 256;
          pattern_suffix += static_cast<unsigned char>(*rev_iterator);
          ++suffix_length;
        }

        while (m_head_start.find(pattern_suffix) == m_head_start.end()) {
          pattern_suffix >>= 8;
          suffix_length--;
        }
        */

        queries.emplace_back(patterns[i], patterns[i].size() - suffix_length, m_head_start[pattern_suffix].first, start_id + i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
        queries.emplace_back(patterns[i], patterns[i].size() - suffix_length, m_head_start[pattern_suffix].second, start_id + i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      }

    } else {
      for (size_t i = 0; i < patterns.size(); ++i) {
        queries.emplace_back(patterns[i], patterns[i].size(), 0, start_id + i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
        queries.emplace_back(patterns[i], patterns[i].size(), m_bwt->global_size(), start_id + i);
        io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      }
    }
    return queries;
  }
};

}  // namespace alx::dist