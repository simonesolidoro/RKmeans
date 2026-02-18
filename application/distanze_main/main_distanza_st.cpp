#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdio.h>
#include <vector>

#include "./../spaziotempo/src/dissimilarities.h"
#include "./../spaziotempo/src/init_policies.h"
#include "./../spaziotempo/src/kmeans_par.h"
//#include "kmeans_seq.h"
//#include "kmeans_old.h"
#include "./../spaziotempo/src/utils.h"

//#include <fdaPDE/fdapde.h>
#include <fdaPDE/models.h>
#include "./fe_separable_par.h"
#include "./fe_separable.h"
#include "./execution.h"
#include <Eigen/Dense>

using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
  std::string output_dir = "./matrice_distanza_st/";
  std::string data_dir = "/work/u10656115/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);



  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // SET PARAMETERS

  // geometry
  Triangulation<3, 3> D("./neuromesh/nodes.csv", "./neuromesh/cells.csv", "./neuromesh/bound.csv", true, true);
  Triangulation<1, 1> T("./neuromesh/REST_f_cpp.csv", true, true);

  // 3D
  FeSpace Vh(D, P1<1>);
  TrialFunction f(Vh);
  TestFunction v(Vh);
  auto a = integral(D)(dot(grad(f), grad(v)));
  ZeroField<3> u;
  auto F = integral(D)(u * v);
  
 auto mass = integral(D)(f * v);
  auto P0 = mass.assemble();
//L2Policy_spaziotempo dist_st(P0,25150,129);
L2Policy_st_ap dist_ap(P0,25150,129);


  
  std::string resp_file = data_dir + "dataset_cntrl_scz.csv";
  Eigen::MatrixXd responses =
      csv2mat<double>(resp_file, 1, 1);


    std::string out_memb_file = output_dir + "/distanza_st_smooth.csv";
    std::ofstream file_memb(out_memb_file);
    if (!file_memb.is_open() ) {
      std::cerr << "Error opening file: " << out_memb_file << std::endl;
      return 1;
    }
    file_memb.close();
    std::cout<<"dati caricati iniziamo calcolo"<<std::endl;
    Eigen::MatrixXd temp_dist;
    temp_dist.resize(responses.rows(), responses.rows());
parallel_for(0, responses.rows(), [&](int i) {
  auto f_i = responses.row(i);
  for (int j = i; j < responses.rows(); ++j) {
    auto f_j = responses.row(j);
    double d = dist_ap(f_i, f_j);
    temp_dist(i, j) = d;
    temp_dist(j, i) = d;
    std::cout<<"(i,j): ("<<i<<","<<j<<") dist = "<<d<<std::endl;
  }
});
    mat2csv(temp_dist, out_memb_file);
  return 0;
}


