
#ifndef INIT_POLICIES_H
#define INIT_POLICIES_H

#include "dissimilarities.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

// Centroids are initialize at random, using seed if specified
struct RandomInitPolicy {
  std::vector<int> init(const Eigen::MatrixXd &Y, Eigen::MatrixXd &centroids,
                        const unsigned k,
                        std::optional<unsigned> seed = std::nullopt) const {
    const unsigned n_obs = Y.rows();

    std::vector<int> obs_ids(n_obs);
    std::iota(obs_ids.begin(), obs_ids.end(), 0);

    // Initialize the random generator, using the seed if specified
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Sample k elements (without replacement)
    std::vector<int> initial_clusters_id;
    initial_clusters_id.reserve(k);
    std::sample(obs_ids.begin(), obs_ids.end(),
                std::back_inserter(initial_clusters_id), k, gen);

    for (unsigned c = 0; c < k; ++c) {
      centroids.row(c) = Y.row(initial_clusters_id[c]);
    }
    return initial_clusters_id;
  }
};

// Init using a manual list of seeds as centroids_id
struct ManualInitPolicy {
  std::vector<int> seeds;

  explicit ManualInitPolicy(const std::vector<int> &s) : seeds(s) {}

  std::vector<int> init(const Eigen::MatrixXd &Y, Eigen::MatrixXd &centroids,
                        const unsigned k) const {
    // Check that the length of the seeds vector is correct
    if (k != seeds.size()) {
      throw std::runtime_error("Seeds size must be exactly k.");
    }

    int n = Y.rows();

    // Check that all the provided seeds are valid
    if (std::ranges::any_of(seeds, [n](int v) { return v < 0 || v >= n; })) {
      throw std::runtime_error("Seed values must be in the range [0, n-1].");
    }

    // Initialize centroids using the seeds
    for (unsigned c = 0; c < k; ++c) {
      centroids.row(c) = Y.row(seeds[c]);
    }
    return seeds;
  }
};

// K-means++ initialization policy
// Implements "k-means++: The Advantages of Careful Seeding",
// doi: 10.5555/1283383.1283494 using the provided dissimilarity (instead of the
// Euclidean used in the paper)
template <typename Dissimilarity> struct KppPolicy {
private:
  const Dissimilarity &sim_;

public:
  explicit KppPolicy(const Dissimilarity &sim) : sim_(sim) {}

  std::vector<int> init(const Eigen::MatrixXd &Y, Eigen::MatrixXd &centroids,
                        unsigned k,
                        std::optional<unsigned> seed = std::nullopt) const {
    if (k > Y.rows()) {
      throw std::runtime_error(
          "k cannot be greater than the number of observations.");
    }
    const int n = Y.rows();

    // Initialize the random generator, using the seed if specified
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Pick the first centroid randomly
    std::uniform_int_distribution<int> unifDist(0, n - 1);
    int first_id = unifDist(gen);
    centroids.row(0) = Y.row(first_id);

    Eigen::VectorXd min_dist_to_c(n);
    for (int i = 0; i < n; ++i) {
      min_dist_to_c[i] = sim_(Y.row(i), centroids.row(0));
    }

    // Keep track of chosen indices (since k is small, a std::vector is ok)
    std::vector<int> c_ids;
    c_ids.reserve(k);
    c_ids.push_back(first_id);

    // Choose each other centroid 1,...,k-1
    for (unsigned c = 1; c < k; ++c) {
      // Force id : c_ids to have 0 distance, so that they do not add weight to
      // the distribution and thus are not selected again
      for (int idx : c_ids) {
        min_dist_to_c[idx] = 0.0;
      }

      Eigen::VectorXd dist_sq = min_dist_to_c.array().square();

      // In-place prefix sum (see report for details)
      std::partial_sum(dist_sq.data(), dist_sq.data() + n, dist_sq.data());

      // Exploiting prefix sum, we have:
      double sum_squared_d = dist_sq(n - 1);

      // Sample from a weighted distribution, such that
      // P(obs) prop to dist_sq(obs)
      std::uniform_real_distribution<double> weighted_distr(0.0, sum_squared_d);
      double r = weighted_distr(gen);

      // Exploit binary search implemented in std::upper_bound
      auto it = std::upper_bound(dist_sq.data(), dist_sq.data() + n, r);
      int id = static_cast<int>(std::distance(dist_sq.data(), it));
      // Degenerate case where the last element is selected
      if (id == n)
        id--;

      // Now that we have the new centroid, we make all the updates:
      centroids.row(c) = Y.row(id);
      c_ids.push_back(id);

      for (int i = 0; i < n; ++i) {
        double d = sim_(Y.row(i), centroids.row(c));
        if (d < min_dist_to_c[i]) {
          min_dist_to_c[i] = d;
        }
      }
    }
    return c_ids;
  }
};

#endif // INIT_POLICIES_H