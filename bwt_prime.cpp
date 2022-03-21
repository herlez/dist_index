#include <filesystem>
#include <fstream>
#include <iostream>

#include "include/util/mio.hpp"

int main(int argc, char **argv) {
  std::filesystem::path sa_path = argv[1];

  if (argc != 2) {
    std::cout << "Format: bwt_prime SA_PATH" << '\n';
    return -1;
  }
  if (!std::filesystem::exists(sa_path)) {
    std::cout << "File " << sa_path << "doesn't exist.\n";
    return -1;
  }

  size_t sa_size = std::filesystem::file_size(sa_path);
  std::error_code error;
  mio::mmap_source mmap;
  mmap.map(sa_path.string(), error);
  if (error) {
    std::cout << "mio error " << error << "\n";
  }

  size_t i = 0;
  for (; i < sa_size / 5; ++i) {
    size_t number = 0;
    for (size_t j = 0; j < 5; ++j) {
      number <<= 8;
      number += mmap[5 * i + (4 - j)];
    }
    if (number == 0) {
      break;
    }
  }

  ++i;
  std::filesystem::path prm_path = sa_path;
  prm_path.replace_extension(".prm");
  std::cout << "Writing " << i << " to " << prm_path << "\n";

  std::ofstream ostr(prm_path, std::ios::binary);
  ostr.write(reinterpret_cast<const char *>(&i), 8);

  float progress = 0.0;
  while (progress < 1.0) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();

    progress += 0.16;  // for demonstration only
  }
  std::cout << std::endl;
}