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
  concatenated_strings(std::vector<std::string> strings) {
    starts.reserve(strings.size());
    for (auto const& str : strings) {
      starts.push_back(str.size());
      std::copy(str.begin(), str.end(), std::back_inserter(concat_string));
    }
    std::exclusive_scan(starts.begin(), starts.end(), starts.begin(), 0);
  }

  size_t size() {
    return starts.size();
  }

  size_t length_all_strings() {
    return concat_string.size();
  }

  std::vector<char>::const_iterator operator[](size_t i) {
    return concat_string.begin() + starts[i];
  }

  std::vector<char> concat_string;
  std::vector<uint32_t> starts;
};

template <typename t_word = tdc::uint40_t>
class bwt_index {
 private:
  alx::bwt const* m_bwt;
  std::vector<alx::rank_query> m_queries;
  std::vector<alx::rank_query> m_queries_buffer;
  std::vector<alx::rank_query> m_finished_queries;
  MPI_Datatype m_query_type = alx::rank_query::mpi_type();
  MPI_Datatype m_query_info_type = alx::rank_query_information::mpi_type();

 public:
  bwt_index() : m_bwt(nullptr) { MPI_Type_commit(&m_query_type); MPI_Type_commit(&m_query_info_type);}

  // Load partial bwt from bwt and primary index file.
  bwt_index(alx::bwt const& bwt) : m_bwt(&bwt) { MPI_Type_commit(&m_query_type); MPI_Type_commit(&m_query_info_type);}

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

  std::vector<size_t> occ_batched(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      // patterns[0] = patterns[16];
      // patterns.resize(1);
      results.reserve(patterns.size());
    }
    // Answer queries in batch
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    alx::io::alxout << "#Patterns=: " << patterns_size << "\n";
    alx::io::alxout << patterns;

    m_finished_queries.clear();
    occ(patterns);

    if (alx::mpi::my_rank() == 0) {
      assert(m_finished_queries.size() == 2 * patterns.size());
      std::sort(m_finished_queries.begin(), m_finished_queries.end(), [](auto const& left, auto const& right) {
        return left.id < right.id;
      });

      for (size_t i = 0; i < m_finished_queries.size(); i += 2) {
        size_t left, right;
        std::tie(left, right) = std::minmax(m_finished_queries[i].border.u64(), m_finished_queries[i + 1].border.u64());
        alx::io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    reset_buffers();
    return results;
  }

  std::vector<size_t> occ_one_by_one(std::vector<std::string> const& patterns) {
    std::vector<size_t> results;
    if (alx::mpi::my_rank() == 0) {
      results.reserve(patterns.size());
    }

    // Answer single queries
    size_t patterns_size = patterns.size();
    MPI_Bcast(&patterns_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    alx::io::alxout << "#Patterns=: " << patterns_size << "\n";

    for (size_t i = 0; i < patterns_size; ++i) {
      std::vector<std::string> single_pattern;

      if (mpi::my_rank() == 0) {
        single_pattern.push_back(patterns[i]);
        alx::io::alxout << "Queries: " << single_pattern << "\n";
      }

      m_finished_queries.clear();
      occ(single_pattern);

      if (alx::mpi::my_rank() == 0) {
        assert(m_finished_queries[0].pos_in_pattern == 0 && m_finished_queries[1].pos_in_pattern == 0);
        assert(m_finished_queries.size() == 2 * single_pattern.size());
        size_t left, right;
        std::tie(left, right) = std::minmax(m_finished_queries[0].border.u64(), m_finished_queries[1].border.u64());
        alx::io::alxout << "Result: " << right - left << "\n";
        results.push_back(right - left);
      }
    }
    reset_buffers();
    return results;
  }

  void occ(std::vector<std::string> const& patterns) {
    initialize_queries(patterns);
    alx::io::alxout << '\n'
                    << m_queries << '\n';

    // Loop : Send and Calculate until finished
    bool global_finished = false;
    while (!global_finished) {
      // Send queries with Alltoallv
      {
        // Sort queries by (are they finished?, border)
        std::sort(m_queries.begin(), m_queries.end(), [](auto const& left, auto const& right) {
          if (left.pos_in_pattern == 0 && right.pos_in_pattern != 0) return true;
          if (left.pos_in_pattern != 0 && right.pos_in_pattern == 0) return false;
          return (left.border < right.border);
        });

        // Define my counts for sending (how many integers do I send to each process?)
        std::vector<int> counts_send(alx::mpi::world_size());
        for (size_t i = 0; i < m_queries.size(); ++i) {
          int target_pe = get_target_pe(m_queries[i]);
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
        m_queries_buffer.resize(num_incoming_queries);

        MPI_Alltoallv(m_queries.data(), counts_send.data(), displacements_send.data(), m_query_type, m_queries_buffer.data(), counts_recv.data(), displacements_recv.data(), m_query_type, MPI_COMM_WORLD);
        std::swap(m_queries, m_queries_buffer);
        // m_queries = m_queries_buffer;

        alx::io::alxout << m_queries << '\n';
      }

      // Calculate
      {
        for (size_t i = 0; i < m_queries.size(); ++i) {
          auto& query = m_queries[i];

          while ((get_target_pe(query) == alx::mpi::my_rank()) && query.pos_in_pattern != 0) {
            query.pos_in_pattern--;
            query.border = m_bwt->next_border(query.border.u64(), query.cur_char());
            alx::io::alxout << "Calculated query[" << i << "]: " << query << "\n";
          }
        }
      }

      update_completed_queries();
      bool local_finished = m_queries.empty();
      MPI_Allreduce(&local_finished, &global_finished, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
  }

  void initialize_queries(std::vector<std::string> const& patterns) {
    alx::io::alxout << "Initializing queries..\n";
    m_queries.resize(patterns.size() * 2);
    m_finished_queries.clear();

    if (alx::mpi::my_rank() == 0) {
      for (size_t i = 0; i < m_queries.size(); ++i) {
        alx::rank_query& query = m_queries[i];
        std::string const& pattern = patterns[i / 2];

        query.id = i / 2;
        std::memcpy(reinterpret_cast<char*>(query.pattern), pattern.c_str(), pattern.size());
        query.pos_in_pattern = pattern.size();
        query.border = (i % 2 == 0) ? 0 : m_bwt->global_size();
        alx::io::alxout << "Initialized query " << i << ": " << query << "\n";
      }
    } else {
      m_queries.resize(0);
    }
  }

  void update_completed_queries() {
    // Completed queries are
    if (alx::mpi::my_rank() == 0) {
      for (size_t i = 0; i < m_queries.size(); ++i) {
        if (m_queries[i].pos_in_pattern == 0) {
          m_finished_queries.push_back(m_queries[i]);

          // Delete completed query by swaping with last element
          m_queries[i] = m_queries.back();
          m_queries.pop_back();
          --i;  // Look at this element again
        }
      }
    }
  }

  int get_target_pe(rank_query const& query) {
    if (query.pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(alx::io::locate_bwt_slice(query.border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
    }
  }
  int get_target_pe(rank_query_information const& query) {
    if (query.pos_in_pattern == 0) {
      return 0;
    } else {
      return std::get<0>(alx::io::locate_bwt_slice(query.border.u64(), m_bwt->global_size(), m_bwt->world_size(), m_bwt->primary_index()));
    }
  }

  void reset_buffers() {
    m_queries = std::vector<alx::rank_query>();
    m_queries_buffer = std::vector<alx::rank_query>();
    m_finished_queries = m_queries = std::vector<alx::rank_query>();
  }
};

}  // namespace alx