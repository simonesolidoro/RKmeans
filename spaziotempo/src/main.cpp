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
#include "kmeans.h"
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
      "Funzione_test"}; 

  std::string output_dir = "../output/";
  std::string data_dir = "../../data/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);

  std::ofstream file_log(output_dir + "log.txt");
  if (!file_log.is_open()) {
    std::cerr << "Error opening log file." << std::endl;
    return 1;
  }
  file_log.close();
  file_log.open(output_dir + "log.txt", std::ios::app);
  if (!file_log.is_open()) {
    std::cerr << "Error reopening log file." << std::endl;
    return 1;
  }

  // read parameters from generation log:
  LogParams params;
  try {
    params = parse_log_scalars("../../data/gen_log_st.txt");
    std::cout << "n_obs_per_clust: " << params.n_obs_per_clust << "\n";
    file_log << "n_obs_per_clust: " << params.n_obs_per_clust << "\n";
    std::cout << "n_clust:         " << params.n_clust << "\n";
    file_log << "n_clust:         " << params.n_clust << "\n";
    std::cout << "N:               " << params.N << "\n";
    file_log << "N:               " << params.N << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error parsing log: " << e.what() << "\n";
    return 1;
  }

  // SET PARAMETERS
  unsigned N = params.N;       // 50; // number of iterations
  unsigned k = params.n_clust; // 3;
  std::size_t n_obs_per_clust = params.n_obs_per_clust; // 10;

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  unsigned max_iter = 25;

  std::optional<unsigned> seed = std::nullopt;
  //seed = std::random_device{}(); // random seed for reproducibility
  seed = params.seed; // 42; // seed for random number generator

  std::optional<std::vector<double>> lambda = std::nullopt;
  // lambda = {1e-8,1e-6}; // regularization parameter for RKMeans

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
  lambda_2d.resize(8,2);

  // grid da popolare con la griglia dei valori da esplorare
  for(int i =0; i<lambda_2d.rows();++i){
      lambda_2d(i,0) = std::pow(10, -9.0 + 0.25 * i);  
      lambda_2d(i,1) = std::pow(10, -7.0 + 0.25 * i);  
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_3d;
  lambda_3d.resize(4,2);

  // grid da popolare con la griglia dei valori da esplorare
  for(int i =0; i<lambda_3d.rows();++i){
      lambda_3d(i,0) = std::pow(10, -8.0 + 0.05 * i);  
      lambda_3d(i,1) = std::pow(10, -6.0 + 0.05 * i);  
  }

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (seed.has_value()) {
    std::cout << "seed:            " << *seed << "\n";
    file_log << "seed:            " << *seed << "\n";
  } else {
    std::cout << "seed:            undefined\n";
    file_log << "seed:            undefined\n";
  }

  // create output directories
//std::cout<<"C1"<<std::endl; 
  for (std::string_view dir1 : {"2d_st", "3d_st"})
    for (std::string_view dir2 : curve_types)
      fs::create_directories(fs::path(output_dir) / dir1 / dir2);
//std::cout<<"C2"<<std::endl; 

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  Eigen::MatrixXd nodes_2d = csv2mat<double>("../../data/2d/nodes.csv");
  Eigen::MatrixXd nodes_3d = csv2mat<double>("../../data/3d/nodes.csv");
//std::cout<<"C3"<<std::endl; 
  Eigen::MatrixXi cells_2d = csv2mat<int>("../../data/2d/cells.csv");
  Eigen::MatrixXi cells_3d = csv2mat<int>("../../data/3d/cells.csv");
//std::cout<<"C4"<<std::endl; 

  Eigen::MatrixXi boundary_nodes_2d = csv2mat<int>("../../data/2d/boundary_nodes.csv");
  Eigen::MatrixXi boundary_nodes_3d = csv2mat<int>("../../data/3d/boundary_nodes.csv");
//std::cout<<"C5"<<std::endl; 

  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);
  Triangulation<3, 3> D3(nodes_3d, cells_3d, boundary_nodes_3d);
//std::cout<<"C6"<<std::endl; 
  Eigen::MatrixXd istanti = csv2mat<double>("../../data/istanti_t.csv");
std::cout<<istanti.rows()<<" "<<istanti.cols()<<std::endl;
  Triangulation<1, 1> T(istanti);
  // PHYSICS
//std::cout<<"C7"<<std::endl; 
  // 2D
  FeSpace Vh_2d(D2, P1<1>);
  TrialFunction f_2d(Vh_2d);
  TestFunction v_2d(Vh_2d);
  auto a_2d = integral(D2)(dot(grad(f_2d), grad(v_2d)));
  ZeroField<2> u_2d;
  auto F_2d = integral(D2)(u_2d * v_2d);
//std::cout<<"C8"<<std::endl; 
  auto R1_2d = a_2d.assemble();
  auto mass_2d = integral(D2)(f_2d * v_2d);
  auto R0_2d = mass_2d.assemble();
//std::cout<<"C9"<<std::endl; 
  // 3D
  FeSpace Vh_3d(D3, P1<1>);
  TrialFunction f_3d(Vh_3d);
  TestFunction v_3d(Vh_3d);
  auto a_3d = integral(D3)(dot(grad(f_3d), grad(v_3d)));
  ZeroField<3> u_3d;
  auto F_3d = integral(D3)(u_3d * v_3d);

//  std::cout<<"creati campp spazio"<<std::endl; 
  auto R1_3d = a_3d.assemble();
  auto mass_3d = integral(D3)(f_3d * v_3d);
  auto R0_3d = mass_3d.assemble();
//std::cout<<"creata massa spazio"<<std::endl; 

  BsSpace Bh(T, 3);   // cubic B-splines in time
  TrialFunction g(Bh);
  TestFunction  w(Bh);
  auto a_T = integral(T)(dxx(g) * dxx(w));
  ZeroField<1> u_T;
  auto F_T = integral(T)(u_T * w);
//std::cout<<"creata campo tempo"<<std::endl; 
  auto mass_time = integral(T)(g*w);
  auto Rt = mass_time.assemble();
//std::cout<<"creata massa tempo"<<std::endl; 
  auto K0_2d = kronecker(Rt,R0_2d);
  auto K0_3d = kronecker(Rt,R0_3d);
//std::cout<<"prodotto Kron per massa spazio tempo fatto"<<std::endl; 
  // PARAMETERS
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  L2Policy dist_2d(K0_2d);
  L2Policy dist_3d(K0_3d);
  // L2Policy_spaziotempo dist_2d(K0_2d,R0_2d.rows(),Rt.rows());
  // L2Policy_spaziotempo dist_3d(K0_3d,R0_3d.rows(),Rt.rows());
  // KppPolicy init_2d(dist_2d);
  // KppPolicy init_3d(dist_3d);
//std::cout<<"creta distanze spazio tempo"<<std::endl; 
  std::vector<int> manual_ids;
  for (std::size_t i = 0; i < k; ++i) {
    manual_ids.push_back(static_cast<int>(i * n_obs_per_clust));
  }
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = n_obs_per_clust * k;

  // CLUSTERING

//std::cout<<"iniziamo"<<std::endl; 
  for (auto &dir1 : {"2d_st", "3d_st"}) { //, "3d_st" per ora solo 2d
    for (unsigned n = 0; n < N; ++n) {
//std::cout<<"crea file mem centr"<<std::endl; 
      std::string out_memb_file = output_dir + dir1 + "/" + curve_types[0] + "/memberships"+ "_" + std::to_string(n) + ".csv";
      std::string out_cent_file = output_dir + dir1 + "/" + curve_types[0] + "/centroids"+ "_" + std::to_string(n) + ".csv";
      std::ofstream file_memb(out_memb_file);
      std::ofstream file_cent(out_cent_file);
      if (!file_memb.is_open() || !file_cent.is_open()) {
        std::cerr << "Error opening file: " << out_memb_file << " or " << out_cent_file << std::endl;
        return 1;
      }
      file_memb.close();
      file_cent.close();
//std::cout<<"carica resp"<<std::endl; 
      std::string resp_file = std::string("../../data/") + dir1 + "/" + curve_types[0] + "/" + curve_types[0] + "_" + std::to_string(n) + ".csv"; // curve_types[0] perche un solo curve type Funzione_test.
      Eigen::MatrixXd responses = csv2mat<double>(resp_file);
      Eigen::MatrixXd responses_smooth(responses.rows(),responses.cols());

      t1 = high_resolution_clock::now();
      //presmoothing 
      if(dir1 == "2d_st"){
//std::cout<<"smoothing resp"<<std::endl; 
        for (int s=0 ; s< responses.rows(); s++){
//std::cout<<"smooth s:"<<s<<std::endl;
          // load data in geoframe
          GeoFrame data(D2, T);
          auto& l1 = data.template insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
          l1.load_vec("y", responses.row(s));
          SRPDE model("y ~ f", data,fe_ls_separable_mono(std::pair {a_2d, F_2d}, std::pair {a_T, F_T})); //perche ricrea modello ogni volta e non fa semplicemente update_response ?              
          // tolot gcv perché crasha 
          GridSearch<2> optimizer;
          optimizer.optimize(fdapde::execution_par, model.gcv_par(100, 476813), lambda_2d);
std::cout<<"ottimo lambda:"<<optimizer.optimum()[0]<<" "<<optimizer.optimum()[1]<<std::endl;
          model.fit(0,optimizer.optimum()[0],optimizer.optimum()[1]);
          auto f_= model.f();
//      std::cout<<"f.cols()"<<f_.cols()<<" r:"<<f_.rows()<<std::endl;
          if(s== 0){
            responses_smooth.resize(responses.rows(),f_.cols()*f_.rows());    
          }
          responses_smooth.row(s)= model.f();
        }
      }
      if(dir1 == "3d_st"){
        for (int s=0 ; s< responses.rows(); s++){
          // load data in geoframe
          GeoFrame data(D3, T);
          auto& l1 = data.template insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
          l1.load_vec("y", responses.row(s));
          SRPDE model("y ~ f", data,fe_ls_separable_mono(std::pair {a_3d, F_3d}, std::pair {a_T, F_T})); //perche ricrea modello ogni volta e non fa semplicemente update_response ?
          GridSearch<2> optimizer;
          optimizer.optimize(fdapde::execution_par, model.gcv_par(100, 476813), lambda_3d);
          model.fit(0,optimizer.optimum()[0],optimizer.optimum()[1]);
          auto f_= model.f();
          if(s== 0){
            responses_smooth.resize(responses.rows(),f_.cols()*f_.rows());    
          }
          responses_smooth.row(s)= model.f();
        }
      }  
      
      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;

    
      if (dir1 == "2d_st") {
        // RKMeans
        // RKMeans_st_parallel_gcv rkmeans(dist_2d, init_manual, D2,T,
        //                 fe_ls_separable_mono(std::pair {a_2d, F_2d}, std::pair {a_T, F_T}), responses_smooth, k,
        //                 max_iter, seed);
        KMeans rkmeans(responses_smooth,dist_2d, init_manual, k,
                        max_iter, seed);
        //rkmeans.set_gcv_grid(lambda_2d);
        // rkmeans.run(lambda);
        rkmeans.run();
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();
      }
      if (dir1 == "3d_st") {
        // RKMeans
        RKMeans_st_parallel_gcv rkmeans(dist_3d, init_manual, D3,T,
                        fe_ls_separable_mono(std::pair {a_3d, F_3d}, std::pair {a_T, F_T}), responses_smooth, k,
                        max_iter, seed);
        rkmeans.set_gcv_grid(lambda_3d);
        rkmeans.run(lambda);
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();
      }


      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);

      std::string kmeans_type = "rkmeans";
      std::ostringstream ss;
      ss << dir1 << "/" << curve_types[0] << "_" << n << ": "
          << kmeans_type << " execution completed in " << n_iter
          << " iterations (max=" << max_iter << "), time (pre-smooth+kmeans):" << elapsed_time;
      std::string msg = ss.str();
      std::cout << msg << std::endl;
      file_log << msg << std::endl;

      Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                          temp_memb.size());
      // append_mat2csv(temp_row_view, out_memb_file);
      // append_mat2csv(temp_centroids, out_cent_file);
      mat2csv(temp_row_view, out_memb_file);
      mat2csv(temp_centroids, out_cent_file);
    }
  }


  
  file_log.close();

  return 0;
}
