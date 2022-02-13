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

namespace alx {

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

class bwt_index {
 private:
  alx::bwt const* m_bwt;

 public:
  bwt_index() : m_bwt(nullptr) {}

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) {}

  /*
    std::vector<size_t> occ_batched_preshared(std::vector<std::string> const& patterns) {
      // Build concatenated string and share between PEs
      concatenated_strings conc_strings(patterns);
      alx::io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
      alx::io::alxout << conc_strings.concat_string << '\n';
      alx::io::alxout << conc_strings.starts <<  '\n';


      //Share conc_string.concat_string
      {
        size_t conc_strings_size = conc_strings.length_all_strings();
        MPI_Bcast(&conc_strings_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
        conc_strings.concat_string.resize(conc_strings_size);
        MPI_Bcast(conc_strings.concat_string.data(), conc_strings.length_all_strings(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
      }
      //Broadcast conc_string.starts
      {
        size_t num_patterns = patterns.size();
        MPI_Bcast(&num_patterns, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
        conc_strings.starts.resize(num_patterns);
        MPI_Bcast(conc_strings.starts.data(), conc_strings.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
      }
      alx::io::alxout << "conc_stings_num=" << conc_strings.size() << " conc_strings_length=" << conc_strings.length_all_strings() << '\n';
      alx::io::alxout << conc_strings.concat_string << '\n';
      alx::io::alxout << conc_strings.starts <<  '\n';
      // Initialize Queries
      std::vector<alx::rank_query_information> queries;
      std::vector<alx::rank_query_information> queries_buffer;
      std::vector<alx::rank_query_information> finished_queries;

      if (alx::mpi::my_rank() == 0) {
        for (size_t i = 0; i < patterns.size(); ++i) {
          alx::rank_query_information query;
          query.id = i;
          query.pos_in_pattern = patterns[i].size();
          query.border = 0;
          queries.push_back(query);
        }
      }
      if (alx::mpi::my_rank() == alx::mpi::world_size()-1) {
        for (size_t i = 0; i < conc_strings.size(); ++i) {
          alx::rank_query_information query;
          query.id = i;
          query.pos_in_pattern = (i==conc_strings.size() -1) ?
                  (conc_strings.length_all_strings() - conc_strings.starts.back()) :
                  (conc_strings.starts[i+1] - conc_strings.starts[i]);
          query.border = m_bwt->global_size();
          queries.push_back(query);
        }
      }


      // Loop: Answer Queries
      bool global_finished = false;
      while (!global_finished) {
        // Send queries with Alltoallv
        {
          // Sort queries by (are they finished?, border)
          std::sort(queries.begin(), queries.end(), [](auto const& left, auto const& right) {
            if (left.pos_in_pattern == 0 && right.pos_in_pattern != 0) return true;
            if (left.pos_in_pattern != 0 && right.pos_in_pattern == 0) return false;
            return (left.border < right.border);
          });

          // Define my counts for sending (how many integers do I send to each process?)
          std::vector<int> counts_send(alx::mpi::world_size(), 0);
          for (size_t i = 0; i < queries.size(); ++i) {
            int target_pe = get_target_pe(queries[i]);
            ++counts_send[target_pe];
          }
          alx::io::alxout << "#SEND: " << counts_send << '\n';

          // Define my counts for receiving (how many integers do I receive from each process?)
          std::vector<int> counts_recv(alx::mpi::world_size());
          //MPI_Barrier(MPI_COMM_WORLD); //remove
          MPI_Alltoall(counts_send.data(), 1, MPI_INT, counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
          alx::io::alxout << "#RECV: " << counts_recv << '\n';

          // Define my displacements for sending (where is located in the buffer each message to send?)
          std::vector<int> displacements_send(alx::mpi::world_size());
          std::exclusive_scan(counts_send.begin(), counts_send.end(), displacements_send.begin(), 0);
          alx::io::alxout << "DSEND: " << displacements_send << '\n';

          // Define my displacements for reception (where to store in buffer each message received?)
          std::vector<int> displacements_recv(alx::mpi::world_size());
          std::exclusive_scan(counts_recv.begin(), counts_recv.end(), displacements_recv.begin(), 0);
          alx::io::alxout << "DRECV: " << displacements_recv << '\n';

          int num_incoming_queries = displacements_recv.back() + counts_recv.back();
          queries_buffer.resize(num_incoming_queries);

          MPI_Alltoallv(queries.data(), counts_send.data(), displacements_send.data(), m_query_info_type, queries_buffer.data(), counts_recv.data(), displacements_recv.data(), m_query_info_type, MPI_COMM_WORLD);
          std::swap(queries, queries_buffer);

          alx::io::alxout << queries << '\n';
        }

        // Calculate
        {
          for (size_t i = 0; i < queries.size(); ++i) {
            auto& query = queries[i];

            while ((get_target_pe(query) == alx::mpi::my_rank()) && query.pos_in_pattern != 0) {
              query.pos_in_pattern--;
              char cur_char = conc_strings[query.id][query.pos_in_pattern];
              query.border = m_bwt->next_border(query.border.u64(), cur_char);
              alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
            }
          }
        }

        // Update completed queries
        if (alx::mpi::my_rank() == 0) {
          for (size_t i = 0; i < queries.size(); ++i) {
            if (queries[i].pos_in_pattern == 0) {
              finished_queries.push_back(queries[i]);

              // Delete completed query by swaping with last element
              queries[i] = queries.back();
              queries.pop_back();
              --i;  // Look at this element again
            }
          }
        }

        bool local_finished = queries.empty();
        MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
      }
      std::vector<size_t> results;
      if (alx::mpi::my_rank() == 0) {
        assert(finished_queries.size() == 2 * patterns.size());
        std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
          return left.id < right.id;
        });

        for (size_t i = 0; i < finished_queries.size(); i += 2) {
          size_t left, right;
          std::tie(left, right) = std::minmax(finished_queries[i].border.u64(), finished_queries[i + 1].border.u64());
          alx::io::alxout << "Result: " << right - left << "\n";
          results.push_back(right - left);
        }
      }
      return results;
    }
    */

    std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
      std::vector<size_t> results;
      results.reserve(patterns.size());
      
      // Answer queries in batch
      size_t patterns_size = patterns.size();
      MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
      alx::io::alxout << "#Patterns=: " << patterns_size << "\n";

      std::vector<alx::rank_query> finished_queries = occ<alx::rank_query>(patterns);

      if (alx::mpi::my_rank() == 0) {
        assert(finished_queries.size() == 2 * patterns.size());
        std::sort(finished_queries.begin(), finished_queries.end(), [](auto const& left, auto const& right) {
          return left.m_id < right.m_id;
        });

        for (size_t i = 0; i < finished_queries.size(); i += 2) {
          size_t left, right;
          std::tie(left, right) = std::minmax(finished_queries[i].m_border.u64(), finished_queries[i + 1].m_border.u64());
          alx::io::alxout << "Result: " << right - left << "\n";
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
    alx::io::alxout << "#Patterns=: " << patterns_size << "\n";

    for (size_t i = 0; i < patterns_size; ++i) {
      std::span single_pattern{patterns.begin() + i, 1};
      if (mpi::my_rank() == 0) {
        alx::io::alxout << "Queries: " << single_pattern << "\n";
      }

      std::vector<alx::rank_query> finished_queries = occ<alx::rank_query>(single_pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(finished_queries[0].pos_in_pattern == 0 && finished_queries[1].pos_in_pattern == 0);
        assert(finished_queries.size() == 2 * single_pattern.size());
        size_t left, right;
        std::tie(left, right) = std::minmax(finished_queries[0].m_border.u64(), finished_queries[1].m_border.u64());
        alx::io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    return results;
  }

  template <typename t_query>
  std::vector<t_query> occ(auto const& patterns, [[maybe_unused]] alx::concatenated_strings const& conc_strings = alx::concatenated_strings()) {
    MPI_Datatype query_mpi_type = t_query::mpi_type();
    MPI_Type_commit(&query_mpi_type);

    std::vector<t_query> queries(initialize_queries<t_query>(patterns));
    std::vector<t_query> queries_buffer;
    std::vector<t_query> finished_queries;
    finished_queries.reserve(patterns.size());

    alx::io::alxout << '\n'
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
        std::vector<int> counts_send(alx::mpi::world_size());
        for (size_t i = 0; i < queries.size(); ++i) {
          int target_pe = get_target_pe(queries[i]);
          ++counts_send[target_pe];
        }
        alx::io::alxout << "#SEND: " << counts_send << '\n';

        // Define my counts for receiving (how many integers do I receive from each process?)
        std::vector<int> counts_recv(alx::mpi::world_size());
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Alltoall(counts_send.data(), 1, MPI_INT, counts_recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
        alx::io::alxout << "#RECV: " << counts_recv << '\n';

        // Define my displacements for sending (where is located in the buffer each message to send?)
        std::vector<int> displacements_send(alx::mpi::world_size());
        std::exclusive_scan(counts_send.begin(), counts_send.end(), displacements_send.begin(), 0);
        alx::io::alxout << "DSEND: " << displacements_send << '\n';

        // Define my displacements for reception (where to store in buffer each message received?)
        std::vector<int> displacements_recv(alx::mpi::world_size());
        std::exclusive_scan(counts_recv.begin(), counts_recv.end(), displacements_recv.begin(), 0);
        alx::io::alxout << "DRECV: " << displacements_recv << '\n';

        int num_incoming_queries = displacements_recv.back() + counts_recv.back();
        queries_buffer.resize(num_incoming_queries);

        MPI_Alltoallv(queries.data(), counts_send.data(), displacements_send.data(), query_mpi_type, queries_buffer.data(), counts_recv.data(), displacements_recv.data(), query_mpi_type, MPI_COMM_WORLD);
        std::swap(queries, queries_buffer);
        alx::io::alxout << queries << '\n';
      }

      // Calculate
      {
        for (size_t i = 0; i < queries.size(); ++i) {
          auto& query = queries[i];

          while ((get_target_pe(query) == alx::mpi::my_rank()) && query.m_pos_in_pattern != 0) {
            query.m_pos_in_pattern--;

            if constexpr (std::is_same<t_query, alx::rank_query>::value) {
              char cur_char = conc_strings[query.m_id][query.m_pos_in_pattern];
              query.m_border = m_bwt->next_border(query.m_border.u64(), cur_char);
            } else if constexpr (std::is_same<t_query, alx::rank_query_information>::value) {
              char cur_char = query.cur_char();
              query.m_border = m_bwt->next_border(query.m_border.u64(), cur_char);
            } else {
              std::cerr << "Unknown query type!\n";
            }

            alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
          }
          // Move answered queries
          if ((alx::mpi::my_rank() == 0) && query.m_pos_in_pattern == 0) {
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
    alx::io::alxout << "Initializing queries..\n";

    for (size_t i = 0; i < patterns.size(); ++i) {
      queries.emplace_back(patterns[i], patterns.size(), 0, i);
      alx::io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
      queries.emplace_back(patterns[i], patterns.size(), m_bwt->global_size(), i);
      alx::io::alxout << "Initialized query " << i << ": " << queries.back() << "\n";
    }
    return queries;
  }

  int get_target_pe(rank_query const& query) {
    if (query.m_pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(alx::io::locate_bwt_slice(query.m_border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
    }
  }
  int get_target_pe(rank_query_information const& query) {
    if (query.m_pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(alx::io::locate_bwt_slice(query.m_border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
    }
  }
};

}  // namespace alx