#ifndef H_UTILS_H
#define H_UTILS_H

#include <Eigen/Dense>
#include <fstream>
#include <regex>
#include <vector>

template <typename T>
concept isEigenMatrix = std::derived_from<T, Eigen::MatrixBase<T>>;

template <typename T>
void vec2csv(const std::vector<T> &vec, const std::string &filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < vec.size(); ++i) {
      file << vec[i] << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file";
  }
}

template <isEigenMatrix T>
void mat2csv(const T &mat, const std::string &filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        file << mat(i, j);
        if (j != mat.cols() - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file" << filename << std::endl;
  }
}

template <typename T>
void binmat2csv(const T &mat, const std::string &filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        file << (mat(i, j) ? "1" : "0");
        if (j != mat.cols() - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file";
  }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
csv2mat(const std::string &filename, bool header = false, bool row_idx = false) {
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  std::vector<T> values;
  std::string line;
  size_t rows = 0, cols = 0;
  bool first_line = true;

  while (std::getline(file, line)) {
    if (header && first_line) {
      first_line = false;
      continue; // skip header row
    }
    first_line = false;

    std::stringstream ss(line);
    std::string cell;
    size_t current_cols = 0;

    // skip first column if row_idx = true
    if (row_idx && std::getline(ss, cell, ',')) {
      // discard
    }

    while (std::getline(ss, cell, ',')) {
      values.push_back((T)std::stod(cell));
      ++current_cols;
    }

    if (cols == 0)
      cols = current_cols;
    else if (cols != current_cols)
      throw std::runtime_error("Inconsistent number of columns in CSV");

    ++rows;
  }

  return Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      values.data(), rows, cols);
}


template <typename T>
void append_mat2csv(const T &mat, const std::string &filename) {
  std::ofstream file(filename, std::ios::app);
  if (file.is_open()) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        file << mat(i, j);
        if (j != mat.cols() - 1) {
          file << ",";
        }
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file" << filename << std::endl;
  }
}

struct LogParams {
  unsigned N = 0;
  unsigned n_clust = 0;
  std::size_t n_obs_per_clust = 0;
  unsigned N_st = 0;
  unsigned n_clust_st = 0;
  std::size_t n_obs_per_clust_st = 0;
  unsigned seed = 0;
};

LogParams parse_log_scalars(const std::string &filename) {
  std::ifstream in(filename);
  if (!in)
    throw std::runtime_error("Cannot open " + filename);

  std::string text{std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>()};

  LogParams params;

  std::regex pattern(R"((n_obs_per_clust|n_clust|N|seed)\s*[:=]\s*(\d+))");
  for (std::sregex_iterator it(text.begin(), text.end(), pattern), end;
       it != end; ++it) {
    const std::string key = (*it)[1];
    const unsigned val = std::stoul((*it)[2]);

    if (key == "n_obs_per_clust")
      params.n_obs_per_clust = val;
    else if (key == "n_clust")
      params.n_clust = val;
    else if (key == "N")
      params.N = val;
    else if (key == "seed")
      params.seed = val;
  }

  return params;
}

void csv2mat_fill(Eigen::MatrixXd &mat, const std::string &filename) {
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  std::string line;
  size_t row = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    size_t col = 0;

    while (std::getline(ss, cell, ',')) {
      if (col >= mat.cols() || row >= mat.rows())
        throw std::runtime_error("Matrix size mismatch in file: " + filename);
      mat(row, col) = std::stod(cell);
      ++col;
    }
    ++row;
  }
}

size_t count_csv_rows(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Could not open file " << filename << "\n";
    return 0;
  }

  size_t count = 0;
  std::string line;
  while (std::getline(file, line)) {
    ++count;
  }

  return count;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
merge_csv2mat(const std::string &filename1, const std::string &filename2,
              bool header = false, bool row_idx = false) {
  auto mat1 = csv2mat<T>(filename1, header, row_idx);
  auto mat2 = csv2mat<T>(filename2, header, row_idx);

  if (mat1.cols() != mat2.cols())
    throw std::runtime_error("CSV column mismatch during merge");

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> merged(
      mat1.rows() + mat2.rows(), mat1.cols());
  merged << mat1, mat2;
  return merged;
}


template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
csv2mat_nuovo(const std::string &filename, bool header = false, bool row_idx = false) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);

  auto trim_inplace = [](std::string &s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  };

  auto strip_quotes_inplace = [](std::string &s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
      s = s.substr(1, s.size() - 2);
    }
  };

  auto parse_cell = [&](std::string cell) -> T {
    trim_inplace(cell);          // rimuove spazi (anche quello iniziale)
    strip_quotes_inplace(cell);  // rimuove "..."
    trim_inplace(cell);          // nel dubbio, dopo lo strip
    if (cell.empty())
      throw std::runtime_error("Empty cell while parsing: " + filename);

    if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(std::stoll(cell));
    } else {
      return static_cast<T>(std::stod(cell));
    }
  };

  std::vector<T> values;
  std::string line;
  size_t rows = 0, cols = 0;
  bool first_line = true;

  while (std::getline(file, line)) {
    if (header && first_line) { first_line = false; continue; }
    first_line = false;

    // skip linee vuote
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    std::stringstream ss(line);
    std::string cell;
    size_t current_cols = 0;

    // skip prima colonna se è indice riga
    if (row_idx && std::getline(ss, cell, ',')) {
      // discard
    }

    while (std::getline(ss, cell, ',')) {
      try {
        values.push_back(parse_cell(cell));
      } catch (const std::exception &e) {
        throw std::runtime_error(
          "CSV parse error in '" + filename + "' at data row " + std::to_string(rows + 1) +
          ", col " + std::to_string(current_cols + 1) + " | cell=" + cell +
          " | " + e.what()
        );
      }
      ++current_cols;
    }

    if (cols == 0) cols = current_cols;
    else if (cols != current_cols)
      throw std::runtime_error("Inconsistent number of columns in CSV: " + filename);

    ++rows;
  }

  return Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  >(values.data(), rows, cols);
}


//mette colonna a 0 se c'è anche solo un NA, (tanto poi in distanza non influisce e tamto NA pochi)
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
csv2mat_nuovo_na_0(const std::string &filename,
                       bool header = false,
                       bool row_idx = false) {

  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);

  auto trim = [](std::string &s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  };

  auto strip_quotes = [](std::string &s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
      s = s.substr(1, s.size() - 2);
  };

  auto is_na = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return (s == "NA");
  };

  auto parse_num = [](const std::string &s) -> T {
    if constexpr (std::is_integral_v<T>)
      return static_cast<T>(std::stoll(s));
    else
      return static_cast<T>(std::stod(s));
  };

  std::vector<std::vector<T>> temp_data;
  std::vector<bool> col_has_na;

  std::string line;
  bool first_line = true;

  while (std::getline(file, line)) {

    if (header && first_line) { first_line = false; continue; }
    first_line = false;

    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    std::stringstream ss(line);
    std::string cell;

    if (row_idx) std::getline(ss, cell, ',');

    std::vector<T> row;
    size_t col_idx = 0;

    while (std::getline(ss, cell, ',')) {

      trim(cell);
      strip_quotes(cell);
      trim(cell);

      if (temp_data.empty())
        col_has_na.push_back(false);

      if (cell.empty() || is_na(cell)) {
        row.push_back(T{});
        col_has_na[col_idx] = true;
      } else {
        try {
          row.push_back(parse_num(cell));
        } catch (...) {
          throw std::runtime_error("CSV parse error in file: " + filename);
        }
      }

      ++col_idx;
    }

    temp_data.push_back(std::move(row));
  }

  if (temp_data.empty())
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();

  size_t rows = temp_data.size();
  size_t cols = temp_data[0].size();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      result(rows, cols);

  for (size_t j = 0; j < cols; ++j) {
    if (col_has_na[j]) {
      result.col(j).setZero();  // 🔥 intera colonna a 0
    } else {
      for (size_t i = 0; i < rows; ++i)
        result(i, j) = temp_data[i][j];
    }
  }

  return result;
}


// se coolonna ha un NA mette uguale a colonna precendente
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
csv2mat_nuovo_na(const std::string &filename,
                 bool header = false,
                 bool row_idx = false) {

  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Cannot open file: " + filename);

  auto trim = [](std::string &s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  };

  auto strip_quotes = [](std::string &s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
      s = s.substr(1, s.size() - 2);
  };

  auto is_na = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return (s == "NA");
  };

  auto parse_num = [](const std::string &s) -> T {
    if constexpr (std::is_integral_v<T>)
      return static_cast<T>(std::stoll(s));
    else
      return static_cast<T>(std::stod(s));
  };

  std::vector<std::vector<T>> temp_data;
  std::vector<bool> col_has_na;

  std::string line;
  bool first_line = true;

  while (std::getline(file, line)) {

    if (header && first_line) { first_line = false; continue; }
    first_line = false;

    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    std::stringstream ss(line);
    std::string cell;

    if (row_idx) std::getline(ss, cell, ',');

    std::vector<T> row;
    size_t col_idx = 0;

    while (std::getline(ss, cell, ',')) {

      trim(cell);
      strip_quotes(cell);
      trim(cell);

      if (temp_data.empty())
        col_has_na.push_back(false);

      if (cell.empty() || is_na(cell)) {
        row.push_back(T{});            // placeholder
        col_has_na[col_idx] = true;    // segna che la colonna ha NA
      } else {
        try {
          row.push_back(parse_num(cell));
        } catch (...) {
          throw std::runtime_error("CSV parse error in file: " + filename);
        }
      }

      ++col_idx;
    }

    temp_data.push_back(std::move(row));
  }

  if (temp_data.empty())
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();

  size_t rows = temp_data.size();
  size_t cols = temp_data[0].size();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      result(rows, cols);

  for (size_t j = 0; j < cols; ++j) {

    if (!col_has_na[j]) {
      // colonna pulita → copia dati originali
      for (size_t i = 0; i < rows; ++i)
        result(i, j) = temp_data[i][j];

    } else {
      // colonna con NA → copia colonna precedente
      if (j == 0) {
        result.col(j).setZero();  // se è la prima, fallback a 0
      } else {
        result.col(j) = result.col(j - 1);
      }
    }
  }

  return result;
}
#endif // H_UTILS_H
