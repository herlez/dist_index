#pragma once

#include <array>
#include <numeric>
#include <string>
#include <tdc/pred/index.hpp>
#include <tdc/uint/uint40.hpp>
#include <vector>
#include <wavelet_tree/wavelet_tree.hpp>

#include "bwt.hpp"
#include "util/spacer.hpp"
#include "util/timer.hpp"


namespace alx {
template <typename t_word = tdc::uint40_t>
class r_index {
 public:
  r_index() {
    for (auto& vec : m_run_lengths) {
      vec.clear();
      vec.push_back(0);
    }
    m_char_sum.fill(0);

    m_bwt_primary_index = 0;
  }

  r_index(std::string const& text) {
    benchutil::timer timer;
    alx::bwt input_bwt(text);
    std::cout << " time_bwt=" << timer.get_and_reset();
    m_bwt_primary_index = input_bwt.primary_index;
    build_structure(input_bwt);
  }

  r_index(alx::bwt const& bwt) : m_bwt_primary_index(bwt.primary_index) {
    build_structure(bwt);
  }

  r_index(std::filesystem::path const& file) {

  }

  size_t text_size() const {
    if(m_run_starts.empty()) { return 0;}

    unsigned char last_run_char = m_run_letters.back();
    auto const& last_char_run_lengths = m_run_lengths[last_run_char];
    size_t last_run_length = last_char_run_lengths.back() - last_char_run_lengths[last_char_run_lengths.size()-2];
    return m_run_starts.back() + last_run_length;
  }
  size_t num_runs() {
    return m_run_starts.size();
  }

  size_t occ(std::string const& pattern, bool debug = false) {
    unsigned char c = pattern.back();
    size_t span_start = m_char_sum[c];
    size_t span_end = m_char_sum[c + 1];
    if (span_start == span_end) {
      return 0;
    }
    if (debug) {
      std::cout << "\nIteration " << pattern.size() - 1 << " after scanning " << c << std::endl;
      std::cout << "Span[" << span_start << ", " << span_end << "]" << std::endl;
    }

    for (size_t i = pattern.size() - 1; i != 0; --i) {
      c = pattern[i - 1];
      size_t rank_start = rank(span_start, c, false);  // no span_start -1, because index F=$xxxxx and L=xxx($)xx and F[i] ~ L[i-1]
      size_t rank_end = rank(span_end, c, false);      // span_end - (span_end > m_bwt_primary_index ? 1 : 0)
      span_start = size_t{m_char_sum[c]} + rank_start;
      span_end = size_t{m_char_sum[c]} + rank_end;
      if (debug) {
        std::cout << "Iteration " << i - 1 << " after scanning " << c << std::endl;
        std::cout << "Rank[" << rank_start << ", " << rank_end << "]" << std::endl;
        std::cout << "Span[" << span_start << ", " << span_end << "]" << std::endl;
      }
      if (span_start == span_end) {
        return 0;
      }
    }
    return span_end - span_start;
  }

  std::vector<size_t> locate_all(std::string const& pattern) {
    return std::vector<size_t>(pattern.length());
  }

  r_index load_from_file(std::string path) { return r_index{}; }
  void save_to_file(std::string path) {}

 private:
  size_t m_bwt_primary_index;
  std::vector<t_word> m_run_starts;          // 5r B
  tdc::pred::Index<t_word> m_pred;           // small enough
  std::vector<unsigned char> m_run_letters;  // 1r B

  using wm_type = decltype(pasta::make_wm<pasta::BitVector>(m_run_letters.begin(), m_run_letters.end(), 256));
  std::unique_ptr<wm_type> m_run_letters_wm;

  //alx::wavelet_tree m_run_letters_wt;

  std::array<std::vector<t_word>, 256> m_run_lengths;  // 5r B
  std::array<t_word, 257> m_char_sum;                  // 5s B

  size_t pred(size_t i) {
    return m_pred.predecessor(m_run_starts.data(), m_run_starts.size(), i).pos;
  }

  size_t run_rank(unsigned char c, size_t i) {
    //return m_run_letters_wt.rank(i, c);
    return m_run_letters_wm->rank(i+1, c);
  }

  // Rank_c(BWT, pos) considering the terminal
  size_t rank(size_t pos, unsigned char c, bool debug = false) {
    if (pos >= m_bwt_primary_index) {
      --pos;
    }
    size_t kth_run = pred(pos);
    size_t run_start = m_run_starts[kth_run];
    size_t num_c_run = run_rank(c, kth_run - 1);

    unsigned char run_symbol = m_run_letters[kth_run];
    if (debug) {
      std::cout << "rank_" << c << "(" << pos << ") = "
                << m_run_lengths[c][num_c_run] + (c == run_symbol ? (pos - run_start + 1) : 0) << std::endl;
      std::cout << "kth_run=" << kth_run << " run_start=" << run_start << " num_c_run=" << num_c_run << std::endl;
    }

    return size_t{m_run_lengths[c][num_c_run]} + (c == run_symbol ? (pos - run_start + 1) : 0);
  }

  void build_structure(alx::bwt const& input_bwt) {
    benchutil::timer timer;
    benchutil::spacer spacer;
    {
      benchutil::timer timer;
      benchutil::spacer spacer;
      // Prepare m_run_length
      for (size_t i = 0; i < 256; ++i) {
        m_run_lengths[i].push_back(0);
      }

      // Build data structure
      for (size_t run_start = 0; run_start < input_bwt.size();) {
        unsigned char run_letter = input_bwt.last_row[run_start];
        m_run_starts.push_back(run_start);
        m_run_letters.push_back(run_letter);

        size_t run_end = run_start + 1;
        while (run_end < input_bwt.size() && input_bwt.last_row[run_end] == run_letter) {
          ++run_end;
        }
        m_run_lengths[run_letter].push_back(m_run_lengths[run_letter].back() + run_end - run_start);
        run_start = run_end;
      }
      m_run_starts.shrink_to_fit();
      for (auto& a : m_run_lengths) {
        a.shrink_to_fit();
      }
      std::cout << " \nruns_time=" << timer.get_and_reset()
                << " runs_mem=" << spacer.get()
                << " runs_mempeak=" << spacer.get_peak();
    }

    // Build predecessor data structure
    {
      benchutil::timer timer;
      benchutil::spacer spacer;

      m_pred = tdc::pred::Index<t_word>(m_run_starts.data(), m_run_starts.size(), 7);
      std::cout << " \npred_time=" << timer.get_and_reset()
                << " pred_mem=" << spacer.get()
                << " pred_mempeak=" << spacer.get_peak();
      // verify_pred();
    }

    // Build wavelet tree
    {
      benchutil::timer timer;
      benchutil::spacer spacer;

      m_run_letters_wm = std::make_unique<wm_type>(m_run_letters.begin(), m_run_letters.end(), 256);
      //m_run_letters_wt = alx::wavelet_tree(m_run_letters);

      std::cout << " \nwt_time=" << timer.get_and_reset()
                << " wt_mem=" << spacer.get()
                << " wt_mempeak=" << spacer.get_peak();
      // verify_wt();
      // verify_rank(input_bwt);
    }

    // Build m_char_sum
    {
      benchutil::timer timer;
      benchutil::spacer spacer;

      for (size_t i = 0; i < m_char_sum.size(); ++i) {
        m_char_sum[i] = 0;
      }
      for (auto c : input_bwt.last_row) {
        ++m_char_sum[c];
      }
      std::exclusive_scan(m_char_sum.begin(), m_char_sum.end(), m_char_sum.begin(), t_word{0});

      std::cout << " carray_time=" << timer.get_and_reset()
                << " carray_mem_peak=" << spacer.get_peak()
                << " carray_mem_ds=" << spacer.get()
                << '\n';
    }

    std::cout << " num_runs=" << m_run_starts.size();
    std::cout << " index_time=" << timer.get_and_reset()
              << " index_mem_peak=" << spacer.get_peak()
              << " index_mem_ds=" << spacer.get()
              << '\n';
  }

  bool verify_pred() {
    benchutil::timer timer;
    if (m_run_starts.empty()) {
      return true;
    }

    if (m_run_starts[0] != 0) {
      std::cerr << "\nFirst entry in m_runs_starts should be 0!" << std::endl;
      return false;
    }
    for (size_t i = 1; i < m_run_starts.size(); ++i) {
      for (size_t j = m_run_starts[i - 1]; j < m_run_starts[i]; ++j) {
        if (pred(j) != i - 1) {
          std::cout << "\nPredecessor of " << j << ": " << pred(j) << " should be " << i - 1 << std::endl;
          return false;
        }
      }
      for (size_t j = m_run_starts.back(); j < text_size(); ++j) {
        if (pred(j) != m_run_starts.size() - 1) {
          std::cout << "\nPredecessor of " << j << ": " << pred(j) << " should be " << m_run_starts.size() - 1 << std::endl;
          return false;
        }
      }
    }
    std::cout << " verify_predecessor=true verify_predecessor_time=" << timer.get() << "\n";
    return true;
  }

  bool verify_wt() {
    benchutil::timer timer;
    std::vector<size_t> histogram(256, 0);

    for (size_t i = 0; i < m_run_letters.size(); ++i) {
      ++histogram[m_run_letters[i]];
      for (size_t c = 0; c < 256; ++c) {
        if (run_rank(c, i) != histogram[c]) {
          std::cout << "Run_rank wrong at   i: " << i << " c:" << c << " Got " << run_rank(c, i) << " but expected " << histogram[c] << '\n';
          return false;
        }
      }
    }

    std::cout << " verify_wt=true verify_wt_time=" << timer.get() << "\n";

    return true;
  }

  bool verify_rank(alx::bwt const& input_bwt) {
    benchutil::timer timer;
    if (input_bwt.size() == 0) {
      return true;
    }

    size_t primary_index = input_bwt.primary_index;
    auto const& last_row = input_bwt.last_row;

    std::vector<size_t> histogram(256, 0);
    for (size_t i = 0; i <= last_row.size(); ++i) {
      if (i < primary_index) {
        ++histogram[last_row[i]];
      }
      if (i > primary_index) {
        ++histogram[last_row[i - 1]];
      }

      for (size_t c = 0; c < 128; ++c) {
        if (rank(i, c) != histogram[c]) {
          std::cout << "Rank_sigma wrong at i: " << i << " c:" << c << " Got " << rank(c, i) << " but expected " << histogram[c] << '\n';
          return false;
        }
      }
    }
    std::cout << " verify_rank=true verify_rank_time=" << timer.get() << "\n";
    return true;
  }
};
}  // namespace alx
