#ifndef KMEANS_H
#define KMEANS_H

#include "dissimilarities.h"
#include "init_policies.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <fdaPDE/fdapde.h>

using namespace fdapde;

inline constexpr unsigned MAX_KMEANS_ITERATIONS = 100;

template <typename DistancePolicy, typename InitPolicy> class KMeans {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;

  std::size_t n_obs_;
  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

public:
  KMeans(const Eigen::MatrixXd &Y, const DistancePolicy &dist,
         const InitPolicy &init_policy, unsigned k = 3,
         unsigned max_iter = MAX_KMEANS_ITERATIONS,
         std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run() {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }
    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }
    }
    /*
    std::cout << " Execution completed in " << n_iter_
              << " iterations (max=" << max_iter_ << ").\n";
    */
  }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};

// RKMeans class for regularized KMeans
template <typename DistancePolicy, typename InitPolicy, typename Triangulation,
          typename Penalty>
class RKMeans {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;
  Triangulation data_;
  std::size_t n_obs_;
  Penalty penalty_;

  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<double> lambda_grid_;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

  void regularize_centroids(std::optional<double> lambda = std::nullopt) {
    for (unsigned c = 0; c < k_; ++c) {
      GeoFrame data(data_);
      auto &l1 = data.template insert_scalar_layer<POINT>("obs", MESH_NODES);
      l1.load_vec("y", centroids_.row(c));

      SRPDE<typename Penalty::solver_t> model("y ~ f", data, penalty_);
      if (lambda) {
        model.fit(0,*lambda);
      } else {
        // calibration
        GridSearch<1> optimizer;
        int seed = (seed_) ? *seed_ : std::random_device{}();
        // if (c == 0) {
        //   optimizer.optimize(model.gcv(100, seed), lambda_grid_);
        //   edf_cache = model.gcv().edf_cache();
        // } else {
        //   optimizer.optimize(model.gcv(edf_cache, 100, seed), lambda_grid_);
        // }
        auto gcv = model.gcv(100, seed);
        optimizer.optimize(gcv, lambda_grid_);
        gcv.edf_cache().clear();
        std::cout << "Optimal lambda for cluster " << c << ": "
                  << optimizer.optimum()[0] << "\n";
        // if (optimizer.optimum()[0] == lambda_grid_.back() ||
        //     optimizer.optimum()[0] == lambda_grid_.front()) {
        //   std::cerr << "Warning: Optimal lambda is at the edge of the grid. "
        //             << "Consider expanding the grid for better results.\n";
        // }
        model.fit(0,optimizer.optimum());
      }

      centroids_.row(c) = model.fitted();
    }
  };

public:
  RKMeans(const DistancePolicy &dist, const InitPolicy &init_policy,
          const Triangulation &triang, const Penalty &penalty,
          const Eigen::MatrixXd &Y, unsigned k = 3,
          unsigned max_iter = MAX_KMEANS_ITERATIONS,
          std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter), data_(triang), penalty_(penalty),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    lambda_grid_.resize(25);
    for (int i = 0; i < 25; ++i) {
      lambda_grid_[i] = std::pow(10, -8.0 + 0.25 * i);
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run(std::optional<double> lambda = std::nullopt) {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual
      // policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }

    // REGULARIZE CENTROIDS
    regularize_centroids(lambda);

    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }

      // REGULARIZE CENTROIDS
      regularize_centroids(lambda);
    }
    // std::cout << " Execution completed in " << n_iter_
    //           << " iterations (max=" << max_iter_ << ").\n";
  }

  void set_gcv_grid(std::vector<double> grid) { lambda_grid_ = grid; }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};



// RKMeans class for regularized KMeans con parallel gcv
template <typename DistancePolicy, typename InitPolicy, typename Triangulation,
          typename Penalty>
class RKMeans_parallel_gcv {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;
  Triangulation data_;
  std::size_t n_obs_;
  Penalty penalty_;

  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<double> lambda_grid_;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

  void regularize_centroids(std::optional<double> lambda = std::nullopt) {
    for (unsigned c = 0; c < k_; ++c) {
      GeoFrame data(data_);
      auto &l1 = data.template insert_scalar_layer<POINT>("obs", MESH_NODES);
      l1.load_vec("y", centroids_.row(c));

      SRPDE<typename Penalty::solver_t> model("y ~ f", data, penalty_); //perche ricrea modello ogni volta e non fa semplicemente update_response ?
      if (lambda) {
        model.fit(0,*lambda);
      } else {
        // calibration
        GridSearch<1> optimizer;
        int seed = (seed_) ? *seed_ : std::random_device{}();
        // if (c == 0) {
        //   optimizer.optimize(model.gcv(100, seed), lambda_grid_);
        //   edf_cache = model.gcv().edf_cache();
        // } else {
        //   optimizer.optimize(model.gcv(edf_cache, 100, seed), lambda_grid_);
        // }
        auto gcv = model.gcv_par(100, seed);// nuovo gcv non sfrutta cache di edf
        optimizer.optimize(fdapde::execution_par,gcv, lambda_grid_);
        gcv.edf_cache().clear(); //che senso ha fare clear ? gcv è oggetto in scope locale di else viene distrutto
        std::cout << "Optimal lambda for cluster " << c << ": "
                  << optimizer.optimum()[0] << "\n";
        // if (optimizer.optimum()[0] == lambda_grid_.back() ||
        //     optimizer.optimum()[0] == lambda_grid_.front()) {
        //   std::cerr << "Warning: Optimal lambda is at the edge of the grid. "
        //             << "Consider expanding the grid for better results.\n";
        // }
        model.fit(0,optimizer.optimum());
      }

      centroids_.row(c) = model.fitted();
    }
  };

public:
  RKMeans_parallel_gcv(const DistancePolicy &dist, const InitPolicy &init_policy,
          const Triangulation &triang, const Penalty &penalty,
          const Eigen::MatrixXd &Y, unsigned k = 3,
          unsigned max_iter = MAX_KMEANS_ITERATIONS,
          std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter), data_(triang), penalty_(penalty),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    lambda_grid_.resize(25);
    for (int i = 0; i < 25; ++i) {
      lambda_grid_[i] = std::pow(10, -8.0 + 0.25 * i);
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run(std::optional<double> lambda = std::nullopt) {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual
      // policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }

    // REGULARIZE CENTROIDS
    regularize_centroids(lambda);

    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }

      // REGULARIZE CENTROIDS
      regularize_centroids(lambda);
    }
    // std::cout << " Execution completed in " << n_iter_
    //           << " iterations (max=" << max_iter_ << ").\n";
  }

  void set_gcv_grid(std::vector<double> grid) { lambda_grid_ = grid; }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};


// RKMeans class for regularized KMeans con parallel gcv per SPAZIO TEMPO
template <typename DistancePolicy, typename InitPolicy, typename Triangulation,typename Triangulationt,
          typename Penalty>
class RKMeans_st_parallel_gcv {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;
  Triangulation data_;
  Triangulationt data_t_;
  std::size_t n_obs_;
  Penalty penalty_;

  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

   Eigen::Matrix<double, Eigen::Dynamic, 2,Eigen::RowMajor> lambda_grid_;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

  void regularize_centroids(std::optional<std::vector<double>> lambda = std::nullopt) {
    for (unsigned c = 0; c < k_; ++c) {
      // load data in geoframe
      GeoFrame data(data_, data_t_);
      auto& l1 = data.template insert_scalar_layer<POINT, POINT>("obs", std::pair {MESH_NODES, MESH_NODES});
      
      l1.load_vec("y", centroids_.row(c));

      SRPDE<typename Penalty::solver_t> model("y ~ f", data, penalty_); //perche ricrea modello ogni volta e non fa semplicemente update_response ?
      if (lambda) {
        model.fit(0,lambda.value()[0],lambda.value()[1]);
      } else {
        // calibration
        GridSearch<2> optimizer;
        int seed = (seed_) ? *seed_ : std::random_device{}();

        auto gcv = model.gcv_par(100, seed);// nuovo gcv non sfrutta cache di edf
        optimizer.optimize(fdapde::execution_par,gcv, lambda_grid_);
        gcv.edf_cache().clear(); //che senso ha fare clear ? gcv è oggetto in scope locale di else viene distrutto
        std::cout << "Optimal lambda for cluster " << c << ": "
                  << optimizer.optimum()[0] <<", "<< optimizer.optimum()[0]<< "\n";
        // if (optimizer.optimum()[0] == lambda_grid_.back() ||
        //     optimizer.optimum()[0] == lambda_grid_.front()) {
        //   std::cerr << "Warning: Optimal lambda is at the edge of the grid. "
        //             << "Consider expanding the grid for better results.\n";
        // }
        model.fit(0,optimizer.optimum()[0],optimizer.optimum()[1]);
      }

      centroids_.row(c) = model.f(); //spazio tempo no fitted() che sono psi*f ma f() che è coeffiecienti di esapnsione fem spline
    }
  };

public:
  RKMeans_st_parallel_gcv(const DistancePolicy &dist, const InitPolicy &init_policy,
          const Triangulation &triang,const Triangulationt &triangt, const Penalty &penalty,
          const Eigen::MatrixXd &Y, unsigned k = 3,
          unsigned max_iter = MAX_KMEANS_ITERATIONS,
          std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter), data_(triang),data_t_(triangt), penalty_(penalty),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    lambda_grid_.resize(25,2);
    // for (int i = 0; i < 25; ++i) {
    //   lambda_grid_[i] = std::pow(10, -8.0 + 0.25 * i);
    // }
    for(int i =0; i<lambda_grid_.rows();++i){
      lambda_grid_(i,0) = std::pow(10, -8.0 + 0.05 * i);  
      lambda_grid_(i,1) = std::pow(10, -8.0 + 0.05 * i);  
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run(std::optional<std::vector<double>> lambda = std::nullopt) {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual
      // policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }

    // REGULARIZE CENTROIDS
    regularize_centroids(lambda);

    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }

      // REGULARIZE CENTROIDS
      regularize_centroids(lambda);
    }
    // std::cout << " Execution completed in " << n_iter_
    //           << " iterations (max=" << max_iter_ << ").\n";
  }

  void set_gcv_grid( Eigen::Matrix<double, Eigen::Dynamic, 2,Eigen::RowMajor> grid) { lambda_grid_ = grid; }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};




#endif // KMEANS_H
