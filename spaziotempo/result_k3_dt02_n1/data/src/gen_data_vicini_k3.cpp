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


  unsigned N_st = 10;
  unsigned k_st = 3;      // due centroidi temporali
  std::size_t n_obs_per_clust_st = 20; //per test solo 4 poi simulazioni vere aumenta a 30
  std::size_t n_obs_st = n_obs_per_clust_st * k_st;
  // noise
  std::mt19937 gen(seed);
  std::normal_distribution<double> noise(0.0, std::sqrt(1));
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0);

  
  if (std::filesystem::exists(output_dir + "2d_st_vicini")) {
    std::filesystem::remove_all(output_dir + "2d_st_vicini");
  }
  std::filesystem::create_directory(output_dir + "2d_st_vicini");


  const std::string wave_type = "vicini";
  // defiizione curva
  // 2D


  // f(x,y,t) con tf = 1.0 
  auto f = [](double x, double y, double tt, double dt) -> double {
      double tf = 1.0;
      double t = tt+dt;
      auto coe = [&](double s) -> double {
          return 0.5 * std::sin(5.0 * pi * s) * std::exp(-s * s) + 1.0;
      };
//sin(2*pi*(coe(y,1)*x*cos(9/5*t/tf-2)-y*sin(9/5*t/tf-2)))*cos(2*pi*(coe(y,1)*x*cos(9/5*t/tf-2+pi/2)+coe(x,1)*y*sin((9/5*t/tf-2)*pi/2)))
      const double theta = (9.0 / 5.0) * (t / tf) - 2.0;
      const double theta2 = (9.0 / 5.0) * (t / tf) - 2.0 + (pi/2.0);

      const double term1 = coe(y) * x * std::cos(theta) - y * std::sin(theta);

      const double term2 = coe(y) * x * (std::cos(theta2)) + coe(x) * y * std::sin( (pi / 2.0) * theta );

      return std::sin(2.0 * pi * term1) * std::cos(2.0 * pi * term2);
  };



  // 
  
  std::filesystem::create_directory(output_dir + "2d_st_vicini" + "/" + wave_type);
  std::filesystem::create_directory(output_dir + "2d_st_vicini" + "/centroidi");
  std::filesystem::create_directory(output_dir + "2d_st_vicini" + "/centroidi/" + wave_type);


  // mesh
  Triangulation<2, 2> D2 = Triangulation<2, 2>::UnitSquare(31);

  // salva mesh
  mat2csv(D2.nodes(), output_dir + "2d_st_vicini/nodes.csv");
  mat2csv(D2.cells(), output_dir + "2d_st_vicini/cells.csv");
  binmat2csv(D2.boundary_nodes(), output_dir + "2d_st_vicini/boundary_nodes.csv");

  auto nodes_2d = D2.nodes();




  // file log per salvare dati generazione
  std::ofstream log_out_st(output_dir + "gen_log_st.txt");
  if (!log_out_st.is_open()) {
    std::cerr << "Error opening gen_log_st.txt for writing.\n";
    return 1;
  }
  log_out_st << "function:         " << wave_type << "\n";
  log_out_st << "combination:      " << " qui non serve la lascio per non cambiare param che sono pigro" << "\n";
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
  unsigned T_nodi = 11;        // numero time steps 
  double sigma_noise = std::sqrt(0.1); // stesso rumore
  

  //mesh
  //temporale 1D 
  Triangulation<1, 1> T = Triangulation<1, 1>::UnitInterval(T_nodi);

  auto nodes_t = T.nodes();   // n_node_T x 1
  mat2csv(T.nodes(), output_dir + "istanti_t.csv");

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
        double dt1 = 0.02; //di quando sono shiftati i due centroidi nel tempo
	double dt2 = 0.04;
        for (unsigned tt = 0; tt < T_nodi; ++tt) {
          double t = T.nodes()(tt,0);

          for (std::size_t j = 0; j < n_nodes; ++j) {
            const double x = nodes_2d(static_cast<int>(j), 0);
            const double y = nodes_2d(static_cast<int>(j), 1);

            const std::size_t idx = static_cast<std::size_t>(tt) * n_nodes + j;

            // salva centroidi una volta
            if (i == 0) {
              cent_no_noise(0, idx) = f(x,y,t,0);
              cent_no_noise(1, idx) = f(x,y,t,dt1);
              cent_no_noise(2, idx) = f(x,y,t,dt2);
            }

            out_st(i, idx)                       = f(x,y,t,0) + noise(gen);
            out_st(i + n_obs_per_clust_st, idx)      = f(x,y,t,dt1) + noise(gen);
            out_st(i +(2* n_obs_per_clust_st), idx)      = f(x,y,t,dt2) + noise(gen);
          }
        }
      }

      mat2csv(cent_no_noise, output_dir + "2d_st_vicini/centroidi/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");

      mat2csv(out_st,        output_dir + "2d_st_vicini/" + wave_type + "/" +
                              wave_type + "_" + std::to_string(n) + ".csv");
    }
  }
  std::cout<<"dati spazio tempo generati"<<std::endl;
  return 0;
}

