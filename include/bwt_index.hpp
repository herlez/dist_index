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

class concatenated_strings {
 public:
  concatenated_strings() = default;

  concatenated_strings(std::vector<std::string> strings) {
    starts.reserve(strings.size());
    for (auto const& str : strings) {
      starts.push_back(str.size());
      std::copy(str.begin(), str.end(), std::back_inserter(concat_string));
    }
    std::exclusive_scan(starts.begin(), starts.end(), starts.begin(), 0);
  }

  size_t size() const {
    return starts.size();
  }

  size_t length_all_strings() const {
    return concat_string.size();
  }

  std::vector<char>::const_iterator operator[](size_t i) const {
    return concat_string.begin() + starts[i];
  }

  std::vector<char> concat_string;
  std::vector<uint32_t> starts;
};

template <typename t_bwt>
class bwt_index {
 private:
  t_bwt const* m_bwt;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt and primary index file.
  bwt_index(t_bwt const& bwt) : m_bwt(&bwt) {}

  std::vector<size_t> occ_batched_preshared(std::vector<std::string> const& patterns) {
    // Build concatenated string and share between PEs
    concatenated_strings conc_strings(patterns);
    io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
    io::alxout << conc_strings.concat_string << '\n';
    io::alxout << conc_strings.starts << '\n';

    // Share conc_string.concat_string
    {
      size_t conc_strings_size = conc_strings.length_all_strings();
      MPI_Bcast(&conc_strings_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
      conc_strings.concat_string.resize(conc_strings_size);
      MPI_Bcast(conc_strings.concat_string.data(), conc_strings.length_all_strings(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    // Broadcast conc_string.starts
    {
      size_t num_patterns = patterns.size();
      MPI_Bcast(&num_patterns, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
      conc_strings.starts.resize(num_patterns);
      MPI_Bcast(conc_strings.starts.data(), conc_strings.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
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

    for (size_t i = 0; i < patterns.size(); ++i) {
      queries.emplace_back(patterns[i], patterns[i].size(), 0, i);
      io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      queries.emplace_back(patterns[i], patterns[i].size(), m_bwt->global_size(), i);
      io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
    }
    return queries;
  }

};

}  // namespace alx