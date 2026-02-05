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
	{  std::string output_dir = "./output_jb_regKmeans_st_noout/";
  std::string data_dir = "/work/u10656115/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);



  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // SET PARAMETERS
  //std::vector<unsigned> k_vec = {2}; // a me interessa solo confronto per speedup teniamo classi 2 e basta.  , 3, 4, 5};
  unsigned k = 2;

  unsigned max_iter = 25;

  unsigned n_runs = 2; // 1 sola run perché mi interessa solo lo speedup

  std::optional<unsigned> seed = std::nullopt;

  seed = std::random_device{}(); // random seed for reproducibility
  std::mt19937_64 rng(*seed);
  //seed = 42; // params.seed; // 42; // seed for random number generator



  std::vector<double> lambda;
  lambda.push_back(std::pow(10, -16.85 ));
  lambda.push_back(std::pow(10, -16.85 ));
//  lambda.push_back(std::pow(10, -6.0 ));
//  lambda.push_back(std::pow(10, -5 ));

  std::cout << "seed: " << *seed << std::endl;
  std::cout << "k: "<< k << std::endl;
  

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // geometry
  Triangulation<3, 3> D("./neuromesh/nodes.csv", "./neuromesh/cells.csv", "./neuromesh/bound.csv", true, true);
  Triangulation<1, 1> T("./neuromesh/REST_f_cpp.csv", true, true);
 // std::cout << D.node(0) << std::endl;
 std::cout << T.n_nodes() << std::endl;

  // 3D
  FeSpace Vh(D, P1<1>);
  TrialFunction f(Vh);
  TestFunction v(Vh);
  auto a = integral(D)(dot(grad(f), grad(v)));
  ZeroField<3> u;
  auto F = integral(D)(u * v);
  
 auto mass = integral(D)(f * v);
  auto P0 = mass.assemble();
  std::cout<<"massa creata"<<std::endl;
L2Policy_spaziotempo dist_st(P0,25150,129);
//L2Policy_st_ap dist_ap(P0,25150,129);
std::cout<<"dist creata"<<std::endl;
  //KppPolicy init(dist_st);
std::vector<std::vector<int>> v_init_runs;
for (int i = 0; i<n_runs; i++){
v_init_runs.push_back({i+30,i+130});
}

std::cout<<"dist e init creati"<<std::endl;

  // CLUSTERING
  unsigned best_idx;
  std::map<unsigned, std::vector<int>> best_memb;
  std::map<unsigned, Eigen::MatrixXd> best_centroids;

  
  std::string resp_file = data_dir + "dataset.csv";
  Eigen::MatrixXd responses =
      csv2mat<double>(resp_file, 0, 0);


std::vector<int> rows_to_keep = {
  0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 14, 16, 18, 20, 21, 22, 23, 24,
  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44,
  45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 64,
  65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
  83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 95, 97, 98, 99, 100, 101, 102, 103,
  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123,
  124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 138, 139, 140, 141, 142, 143, 144,
  145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163,
  164, 165, 166, 167, 168, 169, 170, 172, 173, 174
};


Eigen::MatrixXd responses_clean(rows_to_keep.size(), responses.cols());


for (int i = 0; i < rows_to_keep.size(); ++i) {
	responses_clean.row(i) = responses.row(rows_to_keep[i]);
}

responses = std::move(responses_clean);


std::cout<<"tot oss: "<<responses.rows()<<" dim ogni oss:"<<responses.cols()<<std::endl;



  for (unsigned run = 0; run < n_runs; ++run) {
    ManualInitPolicy init(v_init_runs[run]);
    std::string run_path = output_dir +"st/run_" + std::to_string(run);
    fs::create_directories(run_path);

    std::string out_memb_file = run_path + "/memberships.csv";
    std::string out_cent_file = run_path + "/centroids.csv";
    std::ofstream file_memb(out_memb_file);
    std::ofstream file_cent(out_cent_file);
    if (!file_memb.is_open() || !file_cent.is_open()) {
      std::cerr << "Error opening file: " << out_memb_file << " or "
                << out_cent_file << std::endl;
      return 1;
    }
    file_memb.close();
    file_cent.close();

    unsigned n_iter;
    std::vector<int> temp_memb;
    Eigen::MatrixXd temp_centroids;
//std::cout<<"crea rkmeans"<<std::endl;
    t1 = high_resolution_clock::now();
//RKMeans_parallel
        RKMeans rkmeans(dist_st, init, D,T,
                        fe_ls_separable_parallel(std::pair {a, F}, 500, 1e-9), responses, k,
                         max_iter, seed);
    //rkmeans.set_gcv_grid(lambda_grid); da sistemate
//std::cout<<"rkemans creato, chiamta a run()"<<std::endl; 
    //rkmeans.run(lambda);
    rkmeans.run(lambda);
    n_iter = rkmeans.n_iterations();
    temp_memb = rkmeans.memberships();
    temp_centroids = rkmeans.centroids();

    t2 = high_resolution_clock::now();
    elapsed_time = duration_cast<duration<double>>(t2 - t1);

    std::ostringstream ss;
    ss << "K = " << k << ": run " << run << " completed in " << n_iter
        << " iterations (max=" << max_iter << "), time:" << elapsed_time;
    std::string msg = ss.str();
    std::cout << msg << std::endl;

    std::cout<<"centroids_.rows(): "<<temp_centroids.rows()<<",  centroids_.cols(): "<<temp_centroids.cols()<<std::endl;

    Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                        temp_memb.size());
    append_mat2csv(temp_row_view, out_memb_file);
    append_mat2csv(temp_centroids, out_cent_file);
  
  }
}
  return 0;
}


