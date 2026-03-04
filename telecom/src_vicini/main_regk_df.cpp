// regK con ottimizzazione gcv per ogni smooth di centroide
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
std::cout<<"1"<<std::endl;
  std::string output_dir = "./output_regK_df_k2_init06/";
  std::string data_dir = "/work/u10656115/data_tesista/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);


std::cout<<"2"<<std::endl;

  // SET PARAMETERS
  unsigned k = 2; // 3;
  unsigned max_iter = 25;
  unsigned N = 1; // da farne di piu però cambiando inizializzazione dopo
  std::optional<unsigned> seed = std::nullopt;
  //seed = std::random_device{}(); // random seed for reproducibility
  seed = 42; // seed for random number generator

  std::optional<std::vector<double>> lambda = std::nullopt;
  //lambda = {1.00e-08, 1.00e-07}; // regularization parameter for RKMeans

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
  lambda_2d.resize(16,2);

  // grid da popolare con la griglia dei valori da esplorare
  for(int i =0; i<lambda_2d.rows();++i){
      lambda_2d(i,0) = std::pow(10, -10.0 + 0.25 * i);
      lambda_2d(i,1) = std::pow(10, -10.0 + 0.25 * i);
  }


std::cout<<"3"<<std::endl;


  
  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  Eigen::MatrixXd nodes_2d = csv2mat_nuovo<double>("/work/u10656115/data_tesista/mesh/mesh_nodes.csv",true,true);
  

std::cout<<"4"<<std::endl;
  Eigen::MatrixXi cells_2d = csv2mat_nuovo<int>("/work/u10656115/data_tesista/mesh/mesh_cells.csv",true,true);

std::cout<<"5"<<std::endl;
  Eigen::MatrixXi boundary_nodes_2d = csv2mat_nuovo<int>("/work/u10656115/data_tesista/mesh/mesh_boundary.csv",true,true);


std::cout<<"6"<<std::endl;
  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);

std::cout<<"7"<<std::endl;

  Eigen::MatrixXd istanti = csv2mat_nuovo<double>("/work/u10656115/data_tesista/time_locs.csv",true,true);
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

  L2Policy_spaziotempo dist_2d_st(R0_2d,R0_2d.rows(),istanti.rows());

  std::vector<int> manual_ids = {0,6};
/*  for (std::size_t i = 0; i < k; ++i) {
    manual_ids.push_back(static_cast<int>(i)); // * n_obs_per_clust));
  }
 */
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = 7;

  // CLUSTERING

std::cout<<"creati spazi, dist and init"<<std::endl;


// kmenas regolarizzato su dataset originale (distanza ST)
  {
    for (unsigned n = 0; n < N; ++n) {// N gia fatto
//std::cout<<"crea file mem centr"<<std::endl;
      std::string out_memb_file = output_dir + "/memberships"+ "_" + std::to_string(n) + ".csv";
      std::string out_cent_file = output_dir + "/centroids"+ "_" + std::to_string(n) + ".csv";
      std::ofstream file_memb(out_memb_file);
      std::ofstream file_cent(out_cent_file);
      if (!file_memb.is_open() || !file_cent.is_open()) {
        std::cerr << "Error opening file: " << out_memb_file << " or " << out_cent_file << std::endl;
        return 1;
      }
      file_memb.close();
      file_cent.close();

      Eigen::MatrixXd responses = csv2mat_nuovo<double>("/work/u10656115/data_tesista/y_noNA.csv",true,false);
      t1 = high_resolution_clock::now();

      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;

      std::cout<<"chiamata kmeans gcv"<<std::endl;
        // RKMeans
        int max_it_ = 15;
        RKMeans_gcv rkmeans(dist_2d_st, init_manual, D2,T,
                        fe_ls_separable_parallel(std::pair {a_2d, F_2d}, 500, 1e-9), responses, k,
                         max_it_, seed);
        rkmeans.set_gcv_grid(lambda_2d);
        rkmeans.run(lambda);
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();
      


      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);

      std::cout<<" execution completed in " << n_iter<< " iterations (max=" << max_iter << "), time (reg-kmeans df, distnza ST):" << elapsed_time;

      Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                          temp_memb.size());
      mat2csv(temp_row_view, out_memb_file);
      mat2csv(temp_centroids, out_cent_file);
   
      // Compute WCSS
      double wcss = 0.0;
      for (unsigned i = 0; i < responses.rows(); ++i) {
        wcss += std::pow(
            dist_2d_st(responses.row(i), temp_centroids.row(temp_memb[i])), 2);
      }
      std::cout<<" k = "<<k<<" wcss = "<<wcss<<std::endl;      
    }
  }

  return 0;
}
