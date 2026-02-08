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
	{  std::string output_dir = "./output_jb_regKmeans_st_50ctrlpiulontani/";
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
v_init_runs.push_back({i+5,i+50});
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
0, 1, 3, 7, 8, 9, 10, 13, 15, 17, 18, 19, 23, 27, 28, 29, 35, 37,
38, 39, 48, 50, 52, 53, 54, 56, 57, 63, 67, 68, 69, 71, 78, 80, 82, 87,
89, 90, 93, 94, 96, 100, 101, 102, 103, 108, 109, 113, 119, 121
};
std::cout<<"tenuti dei sani solo "<<rows_to_keep.size()<<std::endl;
for(int i = 0; i<50; i++){
	rows_to_keep.push_back((125+i));
}

std::cout<<"tenuti totali "<<rows_to_keep.size()<<" elenco indici tenuti:"<<std::endl;
for (auto x: rows_to_keep){
	std::cout<<x<<", ";
}
std::cout<<std::endl;
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


