#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <string>

#include "../src/utils.h"
#include <fdaPDE/fdapde.h>
#include <Eigen/Dense>

using std::numbers::pi;
using namespace fdapde;

int main() {
  std::string output_dir = "./";


  //unsigned seed = std::random_device{}();
  unsigned seed = 42;

  std::string combination = "001122";//"021102"; // "001122" or "021102" come combina convessa di tre centroidi (centro fw bw)
  unsigned N = 2;                     // numero dataset generati
  unsigned k = 3;                     // numero gruppi
  std::size_t n_obs_per_clust = 30;
  std::size_t n_obs = n_obs_per_clust * k;

  unsigned N_st = 5;
  unsigned k_st = 2;      // due centroidi temporali
  std::size_t n_obs_per_clust_st = 20; //per test solo 4 poi simulazioni vere aumenta a 30
  std::size_t n_obs_st = n_obs_per_clust_st * k_st;
  // noise
  std::mt19937 gen(seed);
  std::normal_distribution<double> noise(0.0, std::sqrt(0.3));
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0);

  // cartelle per dati 2d e 3d
  for (auto &dir : {"2d", "3d"}) {
    if (std::filesystem::exists(output_dir + dir)) {
      std::filesystem::remove_all(output_dir + dir);
    }
    std::filesystem::create_directory(output_dir + dir);
  }

  const std::string wave_type = "Funzione_test";
  // defiizione curva
  // 2D
  auto Funzione_test_2d_temp = [](double x, double y, double A, double B) {
    const double term1 = std::cos(10.0 * pi * (0.9 * x + 0.3 * y)) * std::exp(-2.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5)));
    const double term2 = A * std::exp(-7.0 * ((x - 0.60)*(x - 0.60) + (y - 0.50)*(y - 0.50)));
    const double term3 = B * std::exp(-13.0 * ((x - 0.20)*(x - 0.20) + (y - 0.30)*(y - 0.30)));
    return term1 - term2 - term3;
  };

  auto Funzione_test_2d = [Funzione_test_2d_temp](double x, double y) {
    return Funzione_test_2d_temp(x,y,1.2,0.7);
  };


  // 3D
  auto Funzione_test_3d = [](double x, double y, double z) {
    const double term1 = std::cos(10.0 * pi * (0.9 * x + 0.3 * y)) * std::exp(-2.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5) + (z - 0.5)*(z - 0.5)));
    const double term2 = 1.2 * std::exp(-7.0 * ((x - 0.60)*(x - 0.60) + (y - 0.50)*(y - 0.50) + (z - 0.55)*(z - 0.55)));
    const double term3 = 0.7 * std::exp(-13.0 * ((x - 0.20)*(x - 0.20) + (y - 0.30)*(y - 0.30) + (z - 0.55)*(z - 0.55)));
    return term1 - term2 - term3;
  };

  // 
  for (auto &dir : {"2d", "3d"}) {
    std::filesystem::create_directory(output_dir + std::string(dir) + "/" + wave_type);
    std::filesystem::create_directory(output_dir + std::string(dir) + "/centroidi");
    std::filesystem::create_directory(output_dir + std::string(dir) + "/centroidi/" + wave_type);
  }

  // mesh
  Triangulation<2, 2> D2 = Triangulation<2, 2>::UnitSquare(31);
  Triangulation<3, 3> D3 = Triangulation<3, 3>::UnitCube(21);

  // salva mesh
  mat2csv(D2.nodes(), output_dir + "2d/nodes.csv");
  mat2csv(D2.cells(), output_dir + "2d/cells.csv");
  binmat2csv(D2.boundary_nodes(), output_dir + "2d/boundary_nodes.csv");

  mat2csv(D3.nodes(), output_dir + "3d/nodes.csv");
  mat2csv(D3.cells(), output_dir + "3d/cells.csv");
  binmat2csv(D3.boundary_nodes(), output_dir + "3d/boundary_nodes.csv");

  auto nodes_2d = D2.nodes();
  auto nodes_3d = D3.nodes();


  // shifts in x di curva 
  auto fw_2d = [Funzione_test_2d](double x, double y) { return Funzione_test_2d(x + 0.1, y); };// 0.1
  auto bw_2d = [Funzione_test_2d](double x, double y) { return Funzione_test_2d(x - 0.1, y); };

  auto fw_3d = [Funzione_test_3d](double x, double y, double z) { return Funzione_test_3d(x + 0.1, y, z); };
  auto bw_3d = [Funzione_test_3d](double x, double y, double z) { return Funzione_test_3d(x - 0.1, y, z); };

  // file log per salvare dati generazione
  std::ofstream log_out(output_dir + "gen_log.txt");
  if (!log_out.is_open()) {
    std::cerr << "Error opening gen_log.txt for writing.\n";
    return 1;
  }
  log_out << "function:         " << wave_type << "\n";
  log_out << "combination:      " << combination << "\n";
  log_out << "n_obs_per_clust:  " << n_obs_per_clust << "\n";
  log_out << "n_clust:          " << k << "\n";
  log_out << "N:                " << N << "\n";
  log_out << "seed:             " << seed << "\n";
  log_out << "u: [\n";
  log_out.close();

  log_out.open(output_dir + "gen_log.txt", std::ios::app);
  if (!log_out.is_open()) {
    std::cerr << "Error opening gen_log.txt for append.\n";
    return 1;
  }

  // generazione dati per ogni dataset n in N
  for (unsigned n = 0; n < N; ++n) {
    const double u = unif_dist(gen);
    log_out << u << "\n";

    // ===== 2D =====
    {
      Eigen::MatrixXd out_no_noise = Eigen::MatrixXd::Zero(k, nodes_2d.rows()); //centroide reale (combo lin di basi) di gruppo, a cui poi aggiunto rumore (salvato per plot)
      Eigen::MatrixXd out          = Eigen::MatrixXd::Zero(n_obs, nodes_2d.rows());

      for (std::size_t i = 0; i < n_obs_per_clust; ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(nodes_2d.rows()); ++j) {
          const double x = nodes_2d(j, 0);
          const double y = nodes_2d(j, 1);

          double c0 = 0.0, c1 = 0.0, c2 = 0.0;

          if (combination == "001122") {
            c0 = u * Funzione_test_2d(x, y) + (1.0 - u) * bw_2d(x, y);
            c1 = u * Funzione_test_2d(x, y) + (1.0 - u) * fw_2d(x, y);
            c2 = u * bw_2d(x, y) + (1.0 - u) * fw_2d(x, y);
          } else if (combination == "021102") {
            c0 = u * Funzione_test_2d(x, y) + (1.0 - u) * bw_2d(x, y);
            c1 = u * fw_2d(x, y) + (1.0 - u) * Funzione_test_2d(x, y);
            c2 = u * bw_2d(x, y) + (1.0 - u) * fw_2d(x, y);
          } else {
            c0 = bw_2d(x, y);
            c1 = Funzione_test_2d(x, y);
            c2 = fw_2d(x, y);
          }
          //salvo centroidi no noise solo una volta
          if(i == 0){
            out_no_noise(0, j) = c0;
            out_no_noise(1, j) = c1;
            out_no_noise(2, j) = c2;
          }  
          out(i, j)                       = c0 + noise(gen);
          out(i + n_obs_per_clust, j)     = c1 + noise(gen);
          out(i + 2 * n_obs_per_clust, j) = c2 + noise(gen);
        }
      }

      mat2csv(out_no_noise, output_dir + "2d/centroidi/" + wave_type + "/" +
                            wave_type + "_" + std::to_string(n) + ".csv");
      mat2csv(out,          output_dir + "2d/" + wave_type + "/" +
                            wave_type + "_" + std::to_string(n) + ".csv");
    }

    // ===== 3D =====
    {
      Eigen::MatrixXd out_no_noise = Eigen::MatrixXd::Zero(n_obs, nodes_3d.rows());
      Eigen::MatrixXd out          = Eigen::MatrixXd::Zero(n_obs, nodes_3d.rows());

      for (std::size_t i = 0; i < n_obs_per_clust; ++i) {
        for (std::size_t j = 0; j < static_cast<std::size_t>(nodes_3d.rows()); ++j) {
          const double x = nodes_3d(j, 0);
          const double y = nodes_3d(j, 1);
          const double z = nodes_3d(j, 2);

          double c0 = 0.0, c1 = 0.0, c2 = 0.0;

          if (combination == "001122") {
            c0 = u * Funzione_test_3d(x, y, z) + (1.0 - u) * bw_3d(x, y, z);
            c1 = u * Funzione_test_3d(x, y, z) + (1.0 - u) * fw_3d(x, y, z);
            c2 = u * bw_3d(x, y, z) + (1.0 - u) * fw_3d(x, y, z);
          } else if (combination == "021102") {
            c0 = u * Funzione_test_3d(x, y, z) + (1.0 - u) * bw_3d(x, y, z);
            c1 = u * fw_3d(x, y, z) + (1.0 - u) * Funzione_test_3d(x, y, z);
            c2 = u * bw_3d(x, y, z) + (1.0 - u) * fw_3d(x, y, z);
          } else {
            c0 = bw_3d(x, y, z);
            c1 = Funzione_test_3d(x, y, z);
            c2 = fw_3d(x, y, z);
          }
          if(i == 0){
            out_no_noise(0, j) = c0;
            out_no_noise(1, j) = c1;
            out_no_noise(2, j) = c2;
          }
          out(i, j)                       = c0 + noise(gen);
          out(i + n_obs_per_clust, j)     = c1 + noise(gen);
          out(i + 2 * n_obs_per_clust, j) = c2 + noise(gen);
        }
      }

      mat2csv(out_no_noise, output_dir + "3d/centroidi/" + wave_type + "/" +
                            wave_type + "_" + std::to_string(n) + ".csv");
      mat2csv(out,          output_dir + "3d/" + wave_type + "/" +
                            wave_type + "_" + std::to_string(n) + ".csv");
    }

    std::cout << "Generated 2D+3D data (Funzione_test) for iteration " << n << "\n";
  }

  log_out << "]\n";
  log_out.close();

  std::cout << "Data spazio generation completed successfully.\n";

  // file log per salvare dati generazione
  std::ofstream log_out_st(output_dir + "gen_log_st.txt");
  if (!log_out_st.is_open()) {
    std::cerr << "Error opening gen_log_st.txt for writing.\n";
    return 1;
  }
  log_out_st << "function:         " << wave_type << "\n";
  log_out_st << "combination:      " << combination << "\n";
  log_out_st << "n_obs_per_clust:  " << n_obs_per_clust_st << "\n";
  log_out_st << "n_clust:          " << k_st << "\n";
  log_out_st << "N:                " << N_st << "\n";
  log_out_st << "seed:             " << seed << "\n";
  log_out_st << "u: [\n";
  log_out_st.close();

  log_out_st.open(output_dir + "gen_log_st.txt", std::ios::app);
  if (!log_out_st.is_open()) {
    std::cerr << "Error opening gen_log_st.txt for append.\n";
    return 1;
  }
  log_out_st << "]\n";
  log_out_st.close();
  //  spazio-tempo separabile 
  unsigned T_nodi = 15;        // numero time steps 
  double sigma_noise = std::sqrt(0.1); // stesso rumore
  auto heaviside = [](double x) { return x >= 0.0 ? 1.0 : 0.0; };

  auto temporal_profile = [heaviside](double t, double t0,double A, double sigma_t,double B, double f, double gamma) {
    const double dt = t - t0;
    // picco gaussiano centrato in t0
    const double peak = A * std::exp(-(dt * dt) / (2.0 * sigma_t * sigma_t));
    // oscillazione smorzata che parte dopo t0
    const double osc  = B * std::sin(2.0 * pi * f * dt) * std::exp(-gamma * dt) * heaviside(dt);
    return peak + osc;
  };

  // due centroidi temporali: picco presto e picco spostato avanti
  double t0_0 = 0.10;
  double Delta_t0 = 0.15;
  double t0_1 = t0_0 + Delta_t0;

  auto g0 = [&](double t) {
    return temporal_profile(t, t0_0,
                            /*A=*/1.0, /*sigma_t=*/0.04,
                            /*B=*/0.6, /*f=*/17.0, /*gamma=*/8.0);
  };

  auto g1 = [&](double t) {
    return temporal_profile(t, t0_1,
                            /*A=*/1.0, /*sigma_t=*/0.04,
                            /*B=*/0.6, /*f=*/17.0, /*gamma=*/8.0);
  };

  //creo cartelle
  for (auto &dir : {"2d_st", "3d_st"}) {
    if (std::filesystem::exists(output_dir + dir)) {
      std::filesystem::remove_all(output_dir + dir);
    }
    std::filesystem::create_directory(output_dir + dir);
    std::filesystem::create_directory(output_dir + std::string(dir) + "/" + wave_type);
    std::filesystem::create_directory(output_dir + std::string(dir) + "/centroidi");
    std::filesystem::create_directory(output_dir + std::string(dir) + "/centroidi/" + wave_type);
  }

  //mesh
  //temporale 1D 
  Triangulation<1, 1> T = Triangulation<1, 1>::UnitInterval(T_nodi);

  auto nodes_t = T.nodes();   // n_node_T x 1
  mat2csv(T.nodes(), output_dir + "istanti_t.csv"); //va aumentata la setprecision in mat2csv perche arrotonda troppo

  //iterazioni 
  for (unsigned n = 0; n<N_st ; n++){
    //2D spazio-tempo separabile 
    {
      const std::size_t n_nodes = static_cast<std::size_t>(nodes_2d.rows());
      const std::size_t p = static_cast<std::size_t>(T_nodi) * n_nodes;
      const std::size_t n_obs_st = n_obs_per_clust_st * k_st;

      Eigen::MatrixXd cent_no_noise = Eigen::MatrixXd::Zero(k_st, p);
      Eigen::MatrixXd out_st        = Eigen::MatrixXd::Zero(n_obs_st, p);

      for (std::size_t i = 0; i < n_obs_per_clust_st; ++i) {

        for (unsigned tt = 0; tt < T_nodi; ++tt) {
          double t = T.nodes()(tt,0);
          const double gt0 = g0(t);
          const double gt1 = g1(t);

          for (std::size_t j = 0; j < n_nodes; ++j) {
            const double x = nodes_2d(static_cast<int>(j), 0);
            const double y = nodes_2d(static_cast<int>(j), 1);

            // spazio uguale per entrambi: base Funzione_test_2d
            const double s = Funzione_test_2d(x, y);

            const std::size_t idx = static_cast<std::size_t>(tt) * n_nodes + j;

            // salva centroidi una volta
            if (i == 0) {
              cent_no_noise(0, idx) = s * gt0;
              cent_no_noise(1, idx) = s * gt1;
            }

            out_st(i, idx)                       = s * gt0 + noise(gen);
            out_st(i + n_obs_per_clust_st, idx)      = s * gt1 + noise(gen);
          }
        }
      }

      mat2csv(cent_no_noise, output_dir + "2d_st/centroidi/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");

      mat2csv(out_st,        output_dir + "2d_st/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");
    }
    // 3D spazio-tempo separabile 
    {
      const std::size_t n_nodes = static_cast<std::size_t>(nodes_3d.rows());
      const std::size_t p = static_cast<std::size_t>(T_nodi) * n_nodes;
      const std::size_t n_obs_st = n_obs_per_clust_st * k_st;

      Eigen::MatrixXd cent_no_noise = Eigen::MatrixXd::Zero(k_st, p);
      Eigen::MatrixXd out_st        = Eigen::MatrixXd::Zero(n_obs_st, p);

      for (std::size_t i = 0; i < n_obs_per_clust_st; ++i) {

        for (unsigned tt = 0; tt < T_nodi; ++tt) {
          double t = T.nodes()(tt,0);

          const double gt0 = g0(t);
          const double gt1 = g1(t);

          for (std::size_t j = 0; j < n_nodes; ++j) {
            const double x = nodes_3d(static_cast<int>(j), 0);
            const double y = nodes_3d(static_cast<int>(j), 1);
            const double z = nodes_3d(static_cast<int>(j), 2);

            const double s = Funzione_test_3d(x, y, z);

            const std::size_t idx = static_cast<std::size_t>(tt) * n_nodes + j;

            if (i == 0) {
              cent_no_noise(0, idx) = s * gt0;
              cent_no_noise(1, idx) = s * gt1;
            }

            out_st(i, idx)                       = s * gt0 + noise(gen);
            out_st(i + n_obs_per_clust_st, idx)      = s * gt1 + noise(gen);
          }
        }
      }

      mat2csv(cent_no_noise, output_dir + "3d_st/centroidi/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");

      mat2csv(out_st,        output_dir + "3d_st/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");
    }

  }
  std::cout<<"dati spazio tempo generati"<<std::endl;
  return 0;
}

