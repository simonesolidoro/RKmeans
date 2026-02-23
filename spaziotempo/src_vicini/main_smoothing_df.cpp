#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdio.h>
#include <vector>

#include "dissimilarities.h"
#include "init_policies.h"
#include "kmeans_par.h"
#include "utils.h"
#include <fdaPDE/fdapde.h>

#include <Eigen/Dense>
#include "./../../application/fe_separable_par.h"
using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
  std::vector<std::string> curve_types = {
      "vicini"};
  std::string data_dir = "./../data/";

  LogParams params;
  try {
    params = parse_log_scalars("./../data/gen_log_st.txt");
    std::cout << "n_obs_per_clust: " << params.n_obs_per_clust << "\n";
    std::cout << "n_clust:         " << params.n_clust << "\n";
    std::cout << "N:               " << params.N << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error parsing log: " << e.what() << "\n";
    return 1;
  }

  // SET PARAMETERS
  unsigned N = params.N;       // 50; // number of iterations
  unsigned k = params.n_clust; // 3;
  std::size_t n_obs_per_clust = params.n_obs_per_clust; // 10;
  unsigned max_iter = 25;
  std::optional<unsigned> seed = std::nullopt;
  //seed = std::random_device{}(); // random seed for reproducibility
  seed = params.seed; // 42; // seed for random number generator


  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
  std::vector<double> ld;
  std::vector<double> lt;
  int size_l = 20;
  for(int i = 0; i<size_l; i++){
  	ld.push_back(std::pow(10, -8 + 0.25 * i));
	lt.push_back(std::pow(10, -8 + 0.25 * i));
  }
  lambda_2d.resize(size_l*size_l,2);
  // grid da popolare con la griglia dei valori da esplorare
  size_t indx = 0;
  for(auto x: ld){
	  for(auto y: lt){
	  	lambda_2d(indx,0) = x;
		lambda_2d(indx,1) = y;
		indx++;
	  }
  }
  std::cout<<"valori lamda:"<<std::endl;
  for(int i = 0; i< size_l*size_l; i++){
  	std::cout<<lambda_2d(i,0)<< " , "<<lambda_2d(i,1)<<std::endl;
  }
  std::cout<<std::endl;

  Eigen::MatrixXd nodes_2d = csv2mat<double>("./../data/2d_st_vicini/nodes.csv");

  Eigen::MatrixXi cells_2d = csv2mat<int>("./../data/2d_st_vicini/cells.csv");

  Eigen::MatrixXi boundary_nodes_2d = csv2mat<int>("./../data/2d_st_vicini/boundary_nodes.csv");


  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);


  Eigen::MatrixXd istanti = csv2mat<double>("./../data/istanti_t.csv");
  Triangulation<1, 1> T(istanti);
  FeSpace Vh_2d(D2, P1<1>);
  TrialFunction f_2d(Vh_2d);
  TestFunction v_2d(Vh_2d);
  auto a_2d = integral(D2)(dot(grad(f_2d), grad(v_2d)));
  ZeroField<2> u_2d;
  auto F_2d = integral(D2)(u_2d * v_2d);



      std::string dirdata = "2d_st_vicini";
      std::string resp_file = std::string("./../data/") + dirdata + "/" + curve_types[0] + "/" + curve_types[0] + "_" + std::to_string(0) + ".csv"; 
     Eigen::MatrixXd responses = csv2mat<double>(resp_file);

          // load data in geoframe
          GeoFrame data(D2, T);
          auto& l1 = data.template insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
          l1.load_vec("y", responses.row(0));
          SRPDE model("y ~ f", data,fe_ls_separable_parallel(std::pair {a_2d, F_2d}, 500, 1e-9)); //perche ricrea modello ogni volta e non fa semplicemente update_response ?
          // tolot gcv per risparmio tempo momentaneo
          GridSearch<2> optimizer;
	  auto m_gcv = model.gcv(100, 476813);
          optimizer.optimize(m_gcv, lambda_2d);
std::cout<<"ottimo lambda:"<<optimizer.optimum()[0]<<" "<<optimizer.optimum()[1]<<std::endl;
auto values = optimizer.values();

std::cout<<"values: "<<std::endl;
for(auto x: values){
	std::cout<<x<<std::endl;
}
    
  

  return 0;
}

