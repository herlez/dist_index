#pragma once

#include <array>
#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "timer.hpp"

namespace alx::io {

std::vector<std::string> read_strings_line_by_line(std::filesystem::path const& path) {
  std::fstream stream(path.c_str(), std::ios::in);
  std::string cur_line;

  std::vector<std::string> result;

  if (stream) {
    while (std::getline(stream, cur_line)) {
      result.push_back(cur_line);
    }
  }
  stream.close();

  return result;
}

std::string load_text(std::filesystem::path const& path, bool timer_output = false) {
  if (!std::filesystem::exists(path)) {
    std::cout << "#FILE " << path << " not found.\n";
    return std::string{};
  }

  benchutil::timer timer;

  std::ifstream t(path);
  t.seekg(0, std::ios::end);
  size_t size = t.tellg();
  std::string buffer(size, ' ');
  t.seekg(0);
  t.read(&buffer[0], size);

  if (timer_output) {
    std::cout << "#READ file from=" << path << " size=" << buffer.size() << " time=" << timer.get_and_reset() << '\n';
  }
  // std::cout << " (" << buffer.size() / 1'000'000 << " MB) in " << timer.get_and_reset() << "ms.\n";
  return buffer;
}

void save_text(std::filesystem::path const& path, std::string const& text, bool timer_output = false) {
  benchutil::timer timer;

  std::ofstream of(path);
  of << text;

  if (timer_output) {
    std::cout << "#WRITE file to=" << path << " size=" << text.size() << " time=" << timer.get_and_reset() << '\n';
  }
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

std::vector<std::string> load_queries(std::filesystem::path const& path) {
  benchutil::timer timer;

  std::vector<std::string> queries;
  std::ifstream ifstr(path);
  std::string line;
  while (std::getline(ifstr, line)) {
    queries.push_back(line);
  }

  std::cout << "#READ queries from=" << path << " size=" << queries.size() << " in=" << timer.get_and_reset() << '\n';
  return queries;
}

std::vector<std::string> generate_queries(std::string const& text, size_t num, size_t length) {
  benchutil::timer timer;

  std::vector<std::string> queries;
  for (size_t i = 0; i < num; ++i) {
    size_t q_pos = i * 100;
    if (q_pos + length >= text.size()) {
      break;
    }
    queries.push_back(text.substr(q_pos, length));
  }
  std::cout << "#GENERATE queries size=" << queries.size() << " time=" << timer.get_and_reset() << '\n';
  return queries;
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

std::vector<std::string> load_patterns(std::filesystem::path path) {
  std::vector<std::string> patterns;
  std::ifstream ifs(path);

  std::string header;
  std::getline(ifs, header);

  size_t n = get_number_of_patterns(header);
  size_t m = get_patterns_length(header);

	//extract patterns from file and search them in the index
	for(size_t i=0;i<n;++i){
		std::string p = std::string();

		for(size_t j=0;j<m;++j){
			char c;
			ifs.get(c);
			p+=c;
		}
    patterns.push_back(p);
	}

  return patterns;
}



}  // namespace alx::io