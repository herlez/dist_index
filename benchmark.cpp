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

  std::vector<size_t> count_results;

 public:
  template <template <typename> typename t_index>
  void run(std::string const& algo) {
    // Load queries
    std::vector<std::string> patterns = alx::io::load_patterns(patterns_path);
    std::cout << "patterns_path=" << patterns_path << " patterns_num=" << patterns.size();
    if (patterns.size() != 0) {
      std::cout << " patterns_len=" << patterns.front().size() << "\n";
    }

    // Load text
    t_index r_index;
    std::string file_name = input_path.filename();

    if (mode == benchmark_mode::from_text) {
      std::err << "Build from index from text not supported yet.\n" return;
    } else if (mode == benchmark_mode::from_bwt) {
      //HERE
      alx::bwt bwt = alx::bwt::load_from_file(input_path);

      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      size_t chunk_size = 
      
      
      std::err << "Build from index from bwt not supported yet.\n" return;



    } else if (mode == benchmark_mode::from_index) {
      std::err << "Build from index from index file not supported yet.\n" return;
    } else {
      std::err << "Unknown benchmark mode. Must be between 0 and 2.\n" return;
    }
    /*
    // Build index
    alx::bwt bwt(bwt_path);



    std::cout << "RESULT"
              << " algo=" << algo
              << " mode=" << mode
              << " text=" << text_name
              << " size=" << text.size();
    alx::benchutil::timer timer;
    alx::benchutil::spacer spacer;
    if (mode == benchmark_mode::from_bwt) {
      alx::bwt bwt(bwt_path);
      r_index = t_index(bwt);
    } else if (mode == benchmark_mode::from_index) {
      // index_path = text_path; //TODO
      // index_path += (algo == "prezza") ? ".ri" : ".rix";
      // r_index = t_index(index_path);
    }

    std::cout << " ds_time" << timer.get()
              << " ds_mem=" << spacer.get()
              << " ds_mempeak=" << spacer.get_peak();
*/

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
  MPI_Init(NULL, NULL);

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
  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name, world_rank, world_size);

  // Finalize the MPI environment.
  MPI_Finalize();
}
