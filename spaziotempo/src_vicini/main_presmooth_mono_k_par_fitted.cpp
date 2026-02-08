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
  std::string output_dir = "./output_presmooth_mono_k_par_fitted/";
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
  lambda = {5.62341e-08, 5.62341e-08}; // regularization parameter for RKMeans

  /* // commentata per test per fare veloce
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
  lambda_2d.resize(16,2);

  // grid da popolare con la griglia dei valori da esplorare
  for(int i =0; i<lambda_2d.rows();++i){
      lambda_2d(i,0) = std::pow(10, -11.0 + 0.25 * i);
      lambda_2d(i,1) = std::pow(10, -11.0 + 0.25 * i);
  }
*/
   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> lambda_2d;
   lambda_2d.resize(1,2);            
   for(int i =0; i<lambda_2d.rows();++i){                                                                                                                          lambda_2d(i,0) = 5.62341e-08;
       lambda_2d(i,1) = 5.62341e-08;
   }

  if (seed.has_value()) {
    std::cout << "seed:            " << *seed << "\n";
  } else {
    std::cout << "seed:            undefined\n";
  }

  fs::create_directories(output_dir+"/2d_st_presmooth_mono_k/vicini");
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

  auto mass_time = integral(T)(g*w);
  auto Rt = mass_time.assemble();
  auto K0_2d = kronecker(Rt,R0_2d);

  L2Policy dist_k(K0_2d);

  std::vector<int> manual_ids;
  for (std::size_t i = 0; i < k; ++i) {
    manual_ids.push_back(static_cast<int>(i)); // * n_obs_per_clust));
  }
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = n_obs_per_clust * k;

  // CLUSTERING

std::cout<<"creati spazi, dist and init"<<std::endl;


// kmenas regolarizzato su dataset originale (distanza ST)
  {
    for (unsigned n = 0; n < N; ++n) {
//std::cout<<"crea file mem centr"<<std::endl;
      std::string out_memb_file = output_dir + "2d_st_presmooth_mono_k/" + curve_types[0] + "/memberships"+ "_" + std::to_string(n) + ".csv";
      std::string out_cent_file = output_dir + "2d_st_presmooth_mono_k/" + curve_types[0] + "/centroids"+ "_" + std::to_string(n) + ".csv";
      // centroidi fitted (non f()) per poter fare plot 
      std::string out_cent_fitted_file = output_dir + "2d_st_presmooth_mono_k/" + curve_types[0] + "/centroids"+ "_" + std::to_string(n) + "fitted.csv";
      std::ofstream file_memb(out_memb_file);
      std::ofstream file_cent(out_cent_file);
      std::ofstream file_cent_fitted(out_cent_fitted_file);
      if (!file_memb.is_open() || !file_cent.is_open() || !file_cent_fitted.is_open()) {
        std::cerr << "Error opening file: " << out_memb_file << " or " << out_cent_file << "o"<<out_cent_fitted_file <<std::endl;
        return 1;
      }
      file_memb.close();
      file_cent.close();
      file_cent_fitted.close();

      std::string dirdata = "2d_st_vicini";
      std::string resp_file = std::string("./../data/") + dirdata + "/" + curve_types[0] + "/" + curve_types[0] + "_" + std::to_string(n) + ".csv"; // curve_types[0] perche un solo curve type Funzione_test.
      Eigen::MatrixXd responses = csv2mat<double>(resp_file);
      t1 = high_resolution_clock::now();

    Eigen::MatrixXd responses_smooth(responses.rows(),responses.cols());
    Eigen::SparseMatrix<double> psi; //per poter passare da f() a fitted()

    for (int s=0 ; s< responses.rows(); s++){
std::cout<<"smooth s:"<<s<<std::endl;
          // load data in geoframe
          GeoFrame data(D2, T);
          auto& l1 = data.template insert_scalar_layer<POINT, POINT>("l1", std::pair {MESH_NODES, MESH_NODES});
          l1.load_vec("y", responses.row(s));
          SRPDE model("y ~ f", data,fe_ls_separable_mono(std::pair {a_2d, F_2d}, std::pair {a_T, F_T})); //perche ricrea modello ogni volta e non fa semplicemente update_response ?
          // tolot gcv per risparmiare tempo
          GridSearch<2> optimizer;
          optimizer.optimize(fdapde::execution_par, model.gcv_par(100, 476813), lambda_2d);
std::cout<<"ottimo lambda:"<<optimizer.optimum()[0]<<" "<<optimizer.optimum()[1]<<std::endl;
          model.fit(0,optimizer.optimum()[0],optimizer.optimum()[1]);
          //model.fit(0,5.62341e-08, 5.62341e-08);
          auto f_= model.f();
          if(s== 0){
            responses_smooth.resize(responses.rows(),f_.cols()*f_.rows());
	    psi = model.get_psi();
          }
          responses_smooth.row(s)= model.f();
    }
      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;
      Eigen::MatrixXd temp_centroids_fitted;
      temp_centroids_fitted.resize(k,psi.rows());

        // RKMeans
        int max_it_ = 15;
        KMeans rkmeans(responses_smooth, dist_k, init_manual, k,
                         max_it_, seed);
        rkmeans.run();
        n_iter = rkmeans.n_iterations();
        temp_memb = rkmeans.memberships();
        temp_centroids = rkmeans.centroids();
	
      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);
	//salvo centroidi fitted
	for(int i = 0; i<k; i++){
	   temp_centroids_fitted.row(i) = psi*temp_centroids.row(i);
	}
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
    
      mat2csv(temp_centroids_fitted, out_cent_fitted_file);
    }
  }

  return 0;
}

