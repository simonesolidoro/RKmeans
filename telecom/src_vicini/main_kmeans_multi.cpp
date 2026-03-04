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
  std::string output_dir = "./output_Kmeansmultivariato/";
  std::string data_dir = "/work/u10656115/data_tesista/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);

  unsigned N = 1;       // 50; // number of iterations
  unsigned k = 2; // 3;
  unsigned max_iter = 25;
  std::optional<unsigned> seed = std::nullopt;
  //seed = std::random_device{}(); // random seed for reproducibility
  seed = 42; // seed for random number generator



  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  Eigen::MatrixXd nodes_2d = csv2mat_nuovo<double>("/work/u10656115/data_tesista/mesh/mesh_nodes.csv",true,true);

  Eigen::MatrixXi cells_2d = csv2mat_nuovo<int>("/work/u10656115/data_tesista/mesh/mesh_cells.csv",true,true);

  Eigen::MatrixXi boundary_nodes_2d = csv2mat_nuovo<int>("/work/u10656115/data_tesista/mesh/mesh_boundary.csv",true,true);


  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);


  Eigen::MatrixXd istanti = csv2mat_nuovo<double>("/work/u10656115/data_tesista/time_locs.csv",true,true);
std::cout<<istanti.rows()<<" "<<istanti.cols()<<std::endl;
  Triangulation<1, 1> T(istanti);
std::cout<<"caricati dati "<<std::endl;

  euclidea dist;

  std::vector<int> manual_ids = {0,6};
 /* for (std::size_t i = 0; i < k; ++i) {
    manual_ids.push_back(static_cast<int>(i)); // * n_obs_per_clust));
  }
 */
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = 7;

  // CLUSTERING

std::cout<<"creati spazi, dist and init"<<std::endl;


// kmenas regolarizzato su dataset originale (distanza ST)
  {
    for (unsigned n = 0; n < N; ++n) {
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

      Eigen::MatrixXd responses = csv2mat<double>("/work/u10656115/data_tesista/y_noNA.csv",true,false);
      t1 = high_resolution_clock::now();

      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;


        // RKMeans
        int max_it_ = 15;
        KMeans_par rkmeans(responses, dist, init_manual, k,
                         max_it_, seed);
        rkmeans.run();
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();

      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);

      std::cout<< " execution completed in " << n_iter
          << " iterations (max=" << max_iter << "), time (multi Kmeans, dist euclidea):" << elapsed_time;

      Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                          temp_memb.size());
  
      mat2csv(temp_row_view, out_memb_file);
      mat2csv(temp_centroids, out_cent_file);
    }
  }

  return 0;
}
