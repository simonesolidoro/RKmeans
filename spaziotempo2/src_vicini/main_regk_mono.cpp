// gcv parallelo (fe_ls_separable_parallel_threaded_exe.h, in qsub va -I fdapde in RKmeans/ non in RKmeans/application/)
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

using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
  std::vector<std::string> curve_types = {
      "vicini"};
  std::string output_dir = "./output_regk_mono_fix1e5_1e4/";
  std::string data_dir = "./../data/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);

  // read parameters from generation log:
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

  std::optional<std::vector<double>> lambda = std::nullopt;
  lambda = {1.00e-05, 1.00e-04}; // regularization parameter for RKMeans

  // commentata per test per fare veloce
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
  lambda_2d.resize(16,2);

  // grid da popolare con la griglia dei valori da esplorare
  for(int i =0; i<lambda_2d.rows();++i){
      lambda_2d(i,0) = std::pow(10, -7.0 + 0.05 * i);
      lambda_2d(i,1) = std::pow(10, -7.0 + 0.05 * i);
  }
/*
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
   lambda_2d.resize(1,2);            
   for(int i =0; i<lambda_2d.rows();++i){                                                                                                                          lambda_2d(i,0) = 5.62341e-08;
       lambda_2d(i,1) = 5.62341e-08;
   }
*/
  if (seed.has_value()) {
    std::cout << "seed:            " << *seed << "\n";
  } else {
    std::cout << "seed:            undefined\n";
  }

  fs::create_directories(output_dir+"/2d_st_regk_mono/vicini");
  std::cout<<"create directory per output"<<std::endl;

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  Eigen::MatrixXd nodes_2d = csv2mat<double>("./../data/2d_st_vicini/nodes.csv");

  Eigen::MatrixXi cells_2d = csv2mat<int>("./../data/2d_st_vicini/cells.csv");

  Eigen::MatrixXi boundary_nodes_2d = csv2mat<int>("./../data/2d_st_vicini/boundary_nodes.csv");


  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);


  Eigen::MatrixXd istanti = csv2mat<double>("./../data/istanti_t.csv");
std::cout<<istanti.rows()<<" "<<istanti.cols()<<std::endl;
  Triangulation<1, 1> T(istanti);
  // PHYSICS
std::cout<<"caricati dati "<<std::endl;
  // 2D
  FeSpace Vh_2d(D2, P1<1>);
  TrialFunction f_2d(Vh_2d);
  TestFunction v_2d(Vh_2d);
  auto a_2d = integral(D2)(dot(grad(f_2d), grad(v_2d)));
  ZeroField<2> u_2d;
  auto F_2d = integral(D2)(u_2d * v_2d);

  auto R1_2d = a_2d.assemble();
  auto mass_2d = integral(D2)(f_2d * v_2d);
  auto R0_2d = mass_2d.assemble();

  //tempo Bspline
  BsSpace Bh(T, 3);   // cubic B-splines in time
  TrialFunction g(Bh);
  TestFunction  w(Bh);
  auto a_T = integral(T)(dxx(g) * dxx(w));
  ZeroField<1> u_T;
  auto F_T = integral(T)(u_T * w);


L2Policy_spaziotempo dist_2d_st(R0_2d,R0_2d.rows(),istanti.rows());
  
  std::vector<int> manual_ids;
  for (std::size_t i = 0; i < k; ++i) {
    //manual_ids.push_back(static_cast<int>(i)); // * n_obs_per_clust));
    manual_ids.push_back(static_cast<int>(i * n_obs_per_clust));
  }
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = n_obs_per_clust * k;

  // CLUSTERING

std::cout<<"creati spazi, dist and init"<<std::endl;


// kmenas regolarizzato su dataset originale (distanza ST)
  {
    for (unsigned n = 0; n < N; ++n) {// al psto che 0 ed N perché voglio rifare solo la 5 con lambda piu fitto
//std::cout<<"crea file mem centr"<<std::endl;
      std::string out_memb_file = output_dir + "2d_st_regk_mono/" + curve_types[0] + "/memberships"+ "_" + std::to_string(n) + ".csv";
      std::string out_cent_file = output_dir + "2d_st_regk_mono/" + curve_types[0] + "/centroids"+ "_" + std::to_string(n) + ".csv";
      std::ofstream file_memb(out_memb_file);
      std::ofstream file_cent(out_cent_file);
      if (!file_memb.is_open() || !file_cent.is_open() ) {
        std::cerr << "Error opening file: " << out_memb_file << " or " << out_cent_file <<std::endl;
        return 1;
      }
      file_memb.close();
      file_cent.close();

      std::string dirdata = "2d_st_vicini";
      std::string resp_file = std::string("./../data/") + dirdata + "/" + curve_types[0] + "/" + curve_types[0] + "_" + std::to_string(n) + ".csv"; // curve_types[0] perche un solo curve type Funzione_test.
      Eigen::MatrixXd responses = csv2mat<double>(resp_file);
      t1 = high_resolution_clock::now();


      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;

        // RKMeans
        int max_it_ = 15;
        RKMeans_gcv_mono rkmeans(dist_2d_st, init_manual,D2,T,fe_ls_separable_mono(std::pair {a_2d, F_2d}, std::pair {a_T, F_T}), responses,k,
                         max_it_, seed);
	rkmeans.set_gcv_grid(lambda_2d);
        rkmeans.run(lambda);
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();
	
      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);
      std::string kmeans_type = "kmeans";
      std::ostringstream ss;
      ss << "2d_st_regk/" << curve_types[0] << "_" << n << ": "
          << kmeans_type << " execution completed in " << n_iter
          << " iterations (max=" << max_iter << "), time (reg-kmeans jb, distnza ST no Kron):" << elapsed_time;
      std::string msg = ss.str();
      std::cout << msg << std::endl;

      Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                          temp_memb.size());

      mat2csv(temp_row_view, out_memb_file);
      mat2csv(temp_centroids, out_cent_file);
    
    }
  }

  return 0;
}

