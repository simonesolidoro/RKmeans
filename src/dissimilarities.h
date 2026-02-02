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
