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

#include <Eigen/Dense>

using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
	  

  std::cout<<"iniziamo"<<std::endl;
  std::string output_dir = "/work/u10656115/";
  std::string data_dir = "/work/u10656115/";
  std::cout<<"set directori data output"<<std::endl;

  std::vector<double> lambda;
  lambda.push_back(std::pow(10, -16.85 ));
  lambda.push_back(std::pow(10, -16.85 ));

  // geometry
  Triangulation<3, 3> D("./neuromesh/nodes.csv", "./neuromesh/cells.csv", "./neuromesh/bound.csv", true, true);
  Triangulation<1, 1> T("./neuromesh/REST_f_cpp.csv", true, true);


  std::cout<<"triangolazione"<<std::endl;
  
  std::string resp_file = data_dir + "dataset.csv";
  Eigen::MatrixXd responses =
      csv2mat<double>(resp_file, 0, 0);

  std::cout<<"caricato dataset"<<std::endl;

int OBS_tot = responses.rows(); 
int OBS_chunk = 5; // 175/5 non devo preoccuparmi di resto
int OBS_start = 0;
//Eigen::MatrixXd responses_smooth;
Eigen::MatrixXd responses_smooth_chunk;
//responses_smooth.resize(OBS_tot, responses.cols());
responses_smooth_chunk.resize(OBS_chunk, responses.cols());
  std::cout<<"resize responses_smooth"<<std::endl;
for(int c = 31; c< (OBS_tot/OBS_chunk); c++){
for (int i = 0; i < OBS_chunk; ++i) 
{
	std::cout<< "inizio smooth i-"<<i<<std::endl;
    GeoFrame data(D, T);
    auto& l1 = data.insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
    l1.load_vec("y", responses.row(i+(c*OBS_chunk)));

    // physics
    FeSpace Vh(D, P1<1>);   // linear finite element in space
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    // laplacian bilinear form
    auto a_D = integral(D)(dot(grad(f), grad(v)));
    ZeroField<3> u_D;
    auto F_D = integral(D)(u_D * v);

    // modeling
    auto t1 = high_resolution_clock::now();
    SRPDE m("y ~ f", data, fe_ls_separable_parallel(std::pair {a_D, F_D}, 500, 1e-9));
    m.fit(lambda[0],lambda[1]);
    Eigen::VectorXd f_smooth = m.fitted();              // forza colonna. uguale a f() tanto fem P1
    responses_smooth_chunk.row(i) = f_smooth.transpose();  // forza riga

//    auto f_smooth = m.fitted(); //uguale a f() tanto fem P1
//    responses_smooth_chunk.row(i)= f_smooth;

	std::cout<< "fine smooth i-"<<i+(c*OBS_chunk)<<std::endl;

}


    std::string out_smooth = output_dir + "/presmoooth_jb_1685_dataset_cntrl_scz_0_124.csv";
    std::ofstream file_smooth(out_smooth, std::ios::app);
    if (!file_smooth.is_open()) {
      std::cerr << "Error opening file: " << out_smooth << std::endl;
      return 1;
    }
    file_smooth.close();

   append_mat2csv(responses_smooth_chunk, out_smooth);
   // non serve rimettere responses_smooth_cunk a 0 perche tanto sovrascritta
}  


  return 0;
}


