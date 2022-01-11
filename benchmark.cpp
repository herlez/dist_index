#include <mpi.h>
#include <stdio.h>

#include <cmdline_parser.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "include/bwt.hpp"
#include "include/r_index_alx.hpp"
#include "include/util/io.hpp"
#include "include/util/spacer.hpp"
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

 public:
  template <template <typename> typename t_index>
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
    alx::io::alxout << "[" << world_rank << "/" << world_size << "]:"
                    << "I am " << processor_name << "\n";

    // Load queries on main node
    std::vector<std::string> patterns;
    if (world_rank == 0) {
      patterns = alx::io::load_patterns(patterns_path);

      alx::io::alxout << "patterns_path=" << patterns_path << " patterns_num=" << patterns.size();
      if (patterns.size() != 0) {
        alx::io::alxout << " patterns_len=" << patterns.front().size() << "\n";
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Build Index
    t_index r_index;
    std::string file_name = input_path.filename();

    alx::benchutil::timer timer;
    alx::benchutil::spacer spacer;
    alx::io::alxout << "RESULT"
                    << " algo=" << algo
                    << " mode=" << mode
                    << " text=" << file_name;

    if (mode == benchmark_mode::from_text) {
      std::cerr << "Build from index from text not supported yet.\n";
      return;
    } else if (mode == benchmark_mode::from_bwt) {
      std::filesystem::path last_row_path = input_path;
      last_row_path += ".bwt";
      std::filesystem::path primary_index_path = input_path;
      primary_index_path += ".prm";
      alx::io::alxout << "[" << world_rank << "/" << world_size << "]: read bwt from " << last_row_path << " and " << primary_index_path << "\n";
      alx::bwt bwt(last_row_path, primary_index_path, world_rank, world_size);
      alx::io::alxout << "[" << world_rank << "/" << world_size << "]: I hold bwt from " << bwt.m_start_index << " to " << bwt.m_end_index << "\n";

      timer.reset();
      spacer.reset();
      r_index = t_index(bwt); //TODO

    } else if (mode == benchmark_mode::from_index) {
      std::cerr << "Build from index from index file not supported yet.\n";
      return;
    } else {
      std::cerr << "Unknown benchmark mode. Must be between 0 and 2.\n";
      return;
    }

    alx::io::alxout << "input_size=" << r_index.text_size()
                    << " ds_time" << timer.get()
                    << " ds_mem=" << spacer.get()
                    << " ds_mempeak=" << spacer.get_peak();

    std::vector<size_t> count_results;
    /*
        // Counting Queries
        timer.reset();
        for (std::string const& q : count_queries) {
          count_results.push_back(r_index.occ(q));
        }
        std::cout << " c_time=" << timer.get()
                  << " c_sum=" << accumulate(count_results.begin(), count_results.end(), 0)
                  << "\n";

        timer.reset();
        */
  }
};

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  index_benchmark benchmark;

  tlx::CmdlineParser cp;
  cp.set_description("Benchmark for text indices");
  cp.set_author("Alexander Herlez <alexander.herlez@tu-dortmund.de>\n");

  std::string input_path;
  cp.add_param_string("input_path", input_path, "Path to the input text/BWT/index");

  std::string patterns_path;
  cp.add_param_string("patterns_path", patterns_path, "Path to pizza&chili patterns");

  unsigned int mode;
  cp.add_param_uint("mode", mode, "Mode: [0]:Text->Index [1]:BWT->Index [2]:Load Index from file");

  if (!cp.process(argc, argv)) {
    std::exit(EXIT_FAILURE);
  }
  benchmark.input_path = input_path;
  benchmark.patterns_path = patterns_path;
  benchmark.mode = static_cast<benchmark_mode>(mode);

  benchmark.run<alx::r_index>("herlez");

  // Finalize the MPI environment.
  MPI_Finalize();
}
