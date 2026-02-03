#ifndef DISSIMILARITIES_H
#define DISSIMILARITIES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

// d(f,g) = sqrt((f-g)^T R0 (f-g))
struct L2Policy {
  Eigen::SparseMatrix<double> R0_;
  L2Policy(const Eigen::SparseMatrix<double> &R0) : R0_(R0) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    // use sparse multiplication
    double squared_norm = diff.transpose() * (R0_ * diff);
    return std::sqrt(squared_norm);
  }
};

// d(f,g) = sqrt(sum_t (f_t-g_t)^T R0 (f_t-g_t))
struct L2Policy_spaziotempo {
  Eigen::SparseMatrix<double> R0_;
  int nodi_spazio;
  int nodi_tempo;

  L2Policy_spaziotempo(const Eigen::SparseMatrix<double> &R0, int ns, int nt)
      : R0_(R0), nodi_spazio(ns), nodi_tempo(nt) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    double distanza = 0.0;

    for (int t = 0; t < nodi_tempo; ++t) {
      auto f_t = f.segment(t * nodi_spazio, nodi_spazio);
      auto g_t = g.segment(t * nodi_spazio, nodi_spazio);

      Eigen::VectorXd diff = f_t - g_t;
      double squared_norm = diff.transpose() * (R0_ * diff);
      distanza += squared_norm;
    }

    return std::sqrt(distanza);
  }
};



// d(f,g) = somma combinazioni f_t e g_i con i da 0 a m e f_ie g_t (f_t-g_t)^T R0 (f_t-g_t))
struct L2Policy_st_ap {
  Eigen::SparseMatrix<double> R0_;
  int nodi_spazio;
  int nodi_tempo;

  L2Policy_st_ap(const Eigen::SparseMatrix<double> &R0, int ns, int nt)
      : R0_(R0), nodi_spazio(ns), nodi_tempo(nt) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    double distanza = 0.0;

    Eigen::VectorXd diff = f - g;
    for (int t = 0; t < nodi_tempo; ++t) {
      auto diff_t = diff.segment(t * nodi_spazio, nodi_spazio);
      for(int tt= 0; tt< nodi_tempo; tt++){
        auto diff_tt = diff.segment(tt * nodi_spazio, nodi_spazio);

        double squared_norm = diff_t.transpose() * (R0_ * diff_tt);
        distanza += squared_norm;
      }
    }

    return distanza; // manca *T ma tanto sarebbe distanze equivalente
  }
};

// d(f,g) = sqrt((f-g)^T R0 (f-g)) / ( sqrt(f^T R0 f) + sqrt(g^T R0 g) )
struct L2NormalizedPolicy {
  Eigen::SparseMatrix<double> R0_;

  L2NormalizedPolicy(const Eigen::SparseMatrix<double> &R0) : R0_(R0) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    double squared_norm = diff.transpose() * (R0_ * diff);
    double f_squared = f.transpose() * (R0_ * f);
    double g_squared = g.transpose() * (R0_ * g);
    double denom = std::sqrt(f_squared) + std::sqrt(g_squared);

    if (denom < 1e-14) {
      return squared_norm;
    }
    return std::sqrt(squared_norm) / denom;
  }
};

// d(f,g) = sqrt((f-g)^T R1 (f-g))
struct R1Policy {
  Eigen::SparseMatrix<double> R1_;

  R1Policy(const Eigen::SparseMatrix<double> &R1) : R1_(R1) {
    if (R1_.rows() != R1_.cols()) {
      throw std::runtime_error("R1 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    double squared_norm = diff.transpose() * (R1_ * diff);
    return std::sqrt(squared_norm);
  }
};

// d(f,g) = sqrt((f-g)^T R1 (f-g)) / ( sqrt(f^T R1 f) + sqrt(g^T R1 g) )
struct NormalizedR1Policy {
  Eigen::SparseMatrix<double> R1_;

  NormalizedR1Policy(const Eigen::SparseMatrix<double> &R1) : R1_(R1) {
    if (R1_.rows() != R1_.cols()) {
      throw std::runtime_error("R1 must be a square matrix!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    double squared_norm = diff.transpose() * (R1_ * diff);
    double f_squared = f.transpose() * (R1_ * f);
    double g_squared = g.transpose() * (R1_ * g);
    double denom = std::sqrt(f_squared) + std::sqrt(g_squared);

    if (denom < 1e-14) {
      return squared_norm;
    }
    return std::sqrt(squared_norm) / denom;
  }
};

// d(f,g) = sqrt((f-g)^T (R0 + R1) (f-g))
struct SobolevPolicy {
  Eigen::SparseMatrix<double> R0_;
  Eigen::SparseMatrix<double> R1_;

  SobolevPolicy(const Eigen::SparseMatrix<double> &R0, const Eigen::SparseMatrix<double> &R1)
      : R0_(R0), R1_(R1) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
    if (R1_.rows() != R1_.cols()) {
      throw std::runtime_error("R1 must be square!");
    }
    if (R0_.rows() != R1_.rows()) {
      throw std::runtime_error("R0 and R1 must have the same dimension!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    // distance = sqrt( diff^T (R0 + R1) diff )
    double squared_norm = diff.transpose() * ((R0_ + R1_) * diff);
    return std::sqrt(squared_norm);
  }
};

// d(f,g) = sqrt((f-g)^T (R0 + R1) (f-g)) / ( sqrt(f^T (R0 + R1) f) + sqrt(g^T
// (R0 + R1) g) )
struct SobolevPolicyNormalized {
  Eigen::SparseMatrix<double> R0_;
  Eigen::SparseMatrix<double> R1_;

  SobolevPolicyNormalized(const Eigen::SparseMatrix<double> &R0, const Eigen::SparseMatrix<double> &R1)
      : R0_(R0), R1_(R1) {
    if (R0_.rows() != R0_.cols()) {
      throw std::runtime_error("R0 must be a square matrix!");
    }
    if (R1_.rows() != R1_.cols()) {
      throw std::runtime_error("R1 must be square!");
    }
    if (R0_.rows() != R1_.rows()) {
      throw std::runtime_error("R0 and R1 must have the same dimension!");
    }
  }

  template <typename T1, typename T2>
  double operator()(const Eigen::MatrixBase<T1> &f,
                    const Eigen::MatrixBase<T2> &g) const {
    Eigen::VectorXd diff = f - g;
    Eigen::SparseMatrix<double> M = R0_ + R1_;

    double squared_norm = diff.transpose() * (M * diff);

    Eigen::VectorXd f_vector = f;
    Eigen::VectorXd g_vector = g;
    double f_squared = f_vector.transpose() * (M * f_vector);
    double g_squared = g_vector.transpose() * (M * g_vector);

    double denom = std::sqrt(f_squared) + std::sqrt(g_squared);
    if (denom < 1e-14) {
      return squared_norm;
    }
    return std::sqrt(squared_norm) / denom;
  }
};

#endif // DISSIMILARITIES_H
