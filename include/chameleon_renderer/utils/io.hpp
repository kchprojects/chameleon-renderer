#pragma once

namespace chameleon {
template <typename T>
void write_vector_bin(std::ofstream& ofs, const std::vector<T>& vec) {
    size_t count = vec.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * count);
}

template <typename T>
std::vector<T> read_vector_bin(std::ifstream& ifs) {
    size_t num_elements = 0;
    ifs.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    std::vector<T> data(num_elements);
    ifs.read(reinterpret_cast<char*>(data.data()), sizeof(T) * num_elements);
    return data;
}

template <typename T>
void read_vector_bin(std::ifstream& ifs,std::vector<T>& data) {
    size_t num_elements = 0;
    ifs.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    data.resize(num_elements);
    ifs.read(reinterpret_cast<char*>(data.data()), sizeof(T) * num_elements);
}

template <typename T>
inline void write_bin(std::ofstream& ofs, const T& var) {
    ofs.write(reinterpret_cast<const char*>(&var), sizeof(var));
}

template <typename T>
inline void read_bin(std::ifstream& ifs, T& var) {
    ifs.read(reinterpret_cast<char*>(&var), sizeof(var));
}

}  // namespace chameleon