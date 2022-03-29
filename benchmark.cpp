#include <mpi.h>
#include <stdio.h>

#include <cmdline_parser.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "include/bwt.hpp"
#include "include/bwt_index.hpp"
#include "include/bwt_rle.hpp"
#include "include/util/io.hpp"
#include "include/util/timer.hpp"

enum benchmark_mode {
  from_text,
  from_bwt,
  from_index
};
class index_benchmark {
 public:
  std::string algo;

  benchmark_mode mode = benchmark_mode::from_bwt;
  std::filesystem::path input_path;
  std::filesystem::path patterns_path;
  size_t num_patterns{std::numeric_limits<size_t>::max()};
  size_t head_start_size = 1'000;
  bool head_start_dynamic = false;

 public:
  template <typename t_bwt, typename t_index>
  void run(std::string const& algo) {
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    alx::dist::io::alxout << "[" << world_rank << "/" << world_size << "]:"
                          << "I am " << processor_name << "\n";

    // Load queries on main node
    std::vector<std::string> patterns;
    {
      // if (world_rank == 0) {
      // alx::dist::benchutil::timer timer;

      patterns = alx::dist::io::load_patterns(patterns_path, num_patterns);
      assert(patterns.size() <= num_patterns);

      /*alx::dist::io::benchout << "patterns_load_time=" << timer.get()
                              << " patterns_path=" << patterns_path
                              << " patterns_num=" << patterns.size();
      if (patterns.size() != 0) {
        alx::dist::io::benchout << " patterns_len=" << patterns.front().size() << "\n";
      }
      */
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Build Index
    t_bwt bwt;
    t_index r_index;
    {
      std::string file_name = input_path.filename();

      alx::dist::io::benchout << "RESULT"
                              << " algo=" << algo
                              << " num_pes=" << world_size
                              << " mode=" << mode
                              << " text=" << file_name;

      alx::dist::benchutil::timer timer;

      if (mode == benchmark_mode::from_text) {
        std::cerr << "Build from text not supported yet.\n";
        return;
      } else if (mode == benchmark_mode::from_bwt) {
        std::filesystem::path last_row_path = input_path;
        last_row_path += ".bwt";

        // alx::dist::io::alxout << "\n[" << world_rank << "/" << world_size << "]: read bwt from " << last_row_path << "\n";
        bwt = t_bwt(last_row_path);
        bwt.build_rank();
        bwt.free_bwt();
        // alx::dist::io::alxout << "[" << world_rank << "/" << world_size << "]: I hold bwt from " << bwt.start_index() << " to " << bwt.end_index() << "\n";
        alx::dist::io::benchout << " bwt_time=" << timer.get_and_reset();

        r_index = t_index(bwt, head_start_dynamic, head_start_size);

      } else if (mode == benchmark_mode::from_index) {
        std::cerr << "Build from index file not supported yet.\n";
        return;
      } else {
        std::cerr << "Unknown benchmark mode. Must be between 0 and 2.\n";
        return;
      }

      alx::dist::io::benchout << " input_size=" << bwt.global_size()
                              << " ds_time=" << timer.get_and_reset();
    }
    // Queries
    std::vector<size_t> count_results;
    {
      alx::dist::benchutil::timer timer;

      // Counting Queries
      timer.reset();
      if (algo == "fm_single" || algo == "r_single") {
        count_results = r_index.occ_one_by_one(patterns);
      }
      if (algo == "fm_batch" || algo == "r_batch") {
        count_results = r_index.occ_batched(patterns);
      }
      if (algo == "fm_batch_preshared" || algo == "r_batch_preshared") {
        count_results = r_index.occ_batched_preshared(patterns);
      }
      // for (auto i: count_results)
      // std::cout << i << ' ';

      if (world_rank == 0) {
        alx::dist::io::benchout << " c_time=" << timer.get_and_reset()
                                << " num_patterns=" << patterns.size()
                                << " c_sum=" << std::accumulate(count_results.begin(), count_results.end(), 0)
                                << "\n";
      } else {
        alx::dist::io::benchout << "\n";
      }
    }
  }
};

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  index_benchmark benchmark;

  tlx::CmdlineParser cp;
  cp.set_description("Benchmark for distributed text indices");
  cp.set_author("Alexander Herlez <alexander.herlez@tu-dortmund.de>\n");

  unsigned int mode;
  cp.add_param_uint("mode", mode, "Mode: [0]:Text->Index [1]:BWT->Index [2]:Load Index from file");

  std::string input_path;
  cp.add_param_string("input_path", input_path, "Path to the input text/BWT/index without file suffix. Enter \"~/text/english\" to load \"~/text/english.bwt\"");

  std::string patterns_path;
  cp.add_param_string("patterns_path", patterns_path, "Path to pizza&chili patterns");

  cp.add_size_t('q', "num_patterns", benchmark.num_patterns,
                "Number of queries (default=all)");

  if (!cp.process(argc, argv)) {
    std::exit(EXIT_FAILURE);
  }
  benchmark.input_path = input_path;
  benchmark.patterns_path = patterns_path;
  benchmark.mode = static_cast<benchmark_mode>(mode);

  // Benchmark different static top trie sizes
  /*{
    for (size_t hs = 1; hs < size_t{2} << 22; hs <<= 2) {
      benchmark.head_start_dynamic = false;
      benchmark.head_start_size = hs;
      benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch");
    }
  }*/
  // Benchmark different dynamic top trie sizes
  /*{
    for (size_t hs = 1; hs < size_t{2} << 22; hs <<= 2) {
      benchmark.head_start_dynamic = true;
      benchmark.head_start_size = hs;
      benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch");
    }
  }*/

  // Benchmark FM-index for different query batch sizes
  {
    size_t old_q_size = benchmark.num_patterns;
    for (size_t q_size = 1; q_size <= old_q_size; q_size *= 2) {
      benchmark.num_patterns = q_size;
      benchmark.run<alx::dist::bwt, alx::dist::bwt_index<alx::dist::bwt>>("fm_batch");
    }
    benchmark.num_patterns = old_q_size;
  }
  {
    size_t old_q_size = benchmark.num_patterns;
    for (size_t q_size = 1; q_size <= old_q_size; q_size *= 2) {
      benchmark.num_patterns = q_size;
      benchmark.run<alx::dist::bwt, alx::dist::bwt_index<alx::dist::bwt>>("fm_batch");
    }
    benchmark.num_patterns = old_q_size;
  }

  // Benchmark r-index for different query batch sizes
  {
    size_t old_q_size = benchmark.num_patterns;
    for (size_t q_size = 1; q_size <= old_q_size; q_size *= 2) {
      benchmark.num_patterns = q_size;
      benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch");
    }
    benchmark.num_patterns = old_q_size;
  }
  {
    size_t old_q_size = benchmark.num_patterns;
    for (size_t q_size = 1; q_size <= old_q_size; q_size *= 2) {
      benchmark.num_patterns = q_size;
      benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch_preshared");
    }
    benchmark.num_patterns = old_q_size;
  }

  // Benchmark FM-Index
  /*benchmark.run<alx::dist::bwt, alx::dist::bwt_index<alx::dist::bwt>>("fm_batch");
  benchmark.run<alx::dist::bwt, alx::dist::bwt_index<alx::dist::bwt>>("fm_batch_preshared");

  // Benchmark r-Index
  benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch");
  benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_batch_preshared");*/

  // deprecated
  // benchmark.run<alx::dist::bwt, alx::dist::bwt_index<alx::dist::bwt>>("fm_single");
  // benchmark.run<alx::dist::bwt_rle, alx::dist::bwt_index<alx::dist::bwt_rle>>("r_single");

  // Finalize the MPI environment.
  MPI_Finalize();
}
