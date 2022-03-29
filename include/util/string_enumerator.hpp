#include <string>
#include <vector>

namespace alx::dist::util {

// Enumerates strings with length up to 8 characters
class string_enumerator {
  std::vector<unsigned char> m_alphabet;     // {A, B, C, H, i}
  std::array<uint8_t, 8> m_cur_string_code;  // {0 0 0 0 0 0 3 4}
  std::string m_cur_string;                  // = "Hi"

 public:
  string_enumerator() {
    for (size_t i = 1; i < 256; ++i) {
      m_alphabet.push_back(i);
    }
    m_cur_string_code.fill(0);
    m_cur_string = "";
  }

  string_enumerator(std::vector<unsigned char> const& alphabet) {
    if(alphabet[0] == 0) {
      m_alphabet = std::vector<unsigned char>(alphabet.begin() + 1, alphabet.end());
    } else {
      m_alphabet = std::vector<unsigned char>(alphabet.begin(), alphabet.end());
    }

    m_cur_string_code.fill(0);
    m_cur_string = "";
  }

  void next() {
    int8_t pos = 7;
    while (m_cur_string_code[pos] == m_alphabet.size()) {
      m_cur_string_code[pos] = 1;
      --pos;
    }
    ++m_cur_string_code[pos];
  }

  std::string get() {
    uint8_t cur_len = current_length();
    m_cur_string.resize(cur_len);
    for (uint8_t pos = 0; pos < cur_len; ++pos) {
      m_cur_string[pos] = m_alphabet[m_cur_string_code[8 - cur_len + pos] -1];
    }
    return m_cur_string;
  }
  size_t get_code() {
    size_t code = 0;
    for (uint8_t pos = 0; pos < 8; ++pos) {
      code <<= 8;
      code += m_cur_string_code[pos];
    }
    return code;
  }

  uint8_t current_length() {
    for (uint8_t pos = 0; pos < 8; ++pos) {
      if (m_cur_string_code[pos] != 0) {
        return 8 - pos;
      }
    }
    return 0;
  }
};

}  // namespace alx::dist::util
