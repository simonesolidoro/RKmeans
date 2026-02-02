// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FE_LS_SEPARABLE_PAR_SOLVER_H__
#define __FE_LS_SEPARABLE_PAR_SOLVER_H__

// #include "header_check.h"

#include "execution.h"

namespace fdapde {
namespace internals {

// central difference time integration loop
class fe_ls_separable_parallel {
   private:
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using diag_matrix_t   = Eigen::DiagonalMatrix<double, Dynamic, Dynamic>;
    using sparse_solver_t = eigen_sparse_solver_movable_wrap<Eigen::SparseLU<sparse_matrix_t>>;
    using dense_solver_t  = Eigen::PartialPivLU<matrix_t>;
    template <typename DataLocs>
    static constexpr bool is_valid_data_locs_descriptor_v =
      std::is_same_v<DataLocs, matrix_t> || std::is_same_v<DataLocs, binary_t>;
    template <typename Penalty> struct is_valid_penalty {
        static constexpr bool value = requires(Penalty penalty) {
            penalty.bilinear_form();
            penalty.linear_form();
            penalty.max_iter();
            penalty.tol();
        };
    };
    template <typename Penalty> static constexpr bool is_valid_penalty_v = is_valid_penalty<Penalty>::value;

    class block_map_t {
        static constexpr int Order = 3;
        using Scalar = double;
        using storage_t = MdMap<Scalar, full_dynamic_extent_t<Order>, internals::layout_left>;

        storage_t data_;
        int rows_ = 0, cols_ = 0;
        int blk_rows_ = 0, blk_cols_ = 0;   // single block size
       public:
        block_map_t() noexcept = default;
        template <typename DataT>
            requires(internals::is_eigen_dense_xpr_v<DataT> &&
                     std::is_same_v<typename std::decay_t<DataT>::Scalar, double>)
        block_map_t(DataT&& data, int rows, int blk_rows, int blk_cols) :
            data_(data.data(), rows, data.cols(), (data.rows() / rows)),
            rows_(rows),
            cols_(data.cols()),
            blk_rows_(blk_rows),
            blk_cols_(blk_cols) {
            fdapde_assert(data.rows() % rows == 0 && rows % blk_rows == 0 && data.cols() % blk_cols == 0);
        }
        template <typename DataT>
            requires(internals::is_eigen_dense_xpr_v<DataT> &&
                     std::is_same_v<typename std::decay_t<DataT>::Scalar, double>)
        block_map_t(DataT&& data, int rows) :   // divide data in (data.rows() / rows) blocks of size rows x data.cols()
            data_(data.data(), rows, data.cols(), (data.rows() / rows)),
            rows_(rows),
	    cols_(data.cols()),
            blk_rows_(rows),
            blk_cols_(data.cols()) {
            fdapde_assert(data.rows() % rows == 0);
        }
        block_map_t(const block_map_t& other) :
            rows_(other.rows_), cols_(other.cols_), blk_rows_(other.blk_rows_), blk_cols_(other.blk_cols_) {
            for (size_t i = 0; i < data_.size(); ++i) { data_.data()[i] = other.data_.data()[i]; }
        }
        block_map_t& operator=(const block_map_t& other) {
            for (size_t i = 0; i < data_.size(); ++i) { data_.data()[i] = other.data_.data()[i]; }
            rows_ = other.rows_;
            cols_ = other.cols_;
            blk_rows_ = other.blk_rows_;
            blk_cols_ = other.blk_cols_;
            return *this;
        }
        // observers
        auto operator()(int i, int k) const {   // get i-th row-block of k-th time instant
            auto slice_ = data_.template slice<2>(k);
            return slice_.as_eigen_map().block(i * blk_rows_, 0, blk_rows_, cols_);
        }
        auto topRows(int j, int k) const {   // get the first j row-blocks of the k-th time instant
            return data_.template slice<2>(k).as_eigen_map().block(0, 0, j * blk_rows_, cols_);
        }
        auto operator()(int k) const { return data_.template slice<2>(k).as_eigen_map(); }
        int size() const { return data_.size(); }
        // modifiers
        auto operator()(int i, int k) {
            auto slice_ = data_.template slice<2>(k);
            return slice_.as_eigen_map().block(i * blk_rows_, 0, blk_rows_, cols_);
        }
        auto topRows(int j, int k) {
            return data_.template slice<2>(k).as_eigen_map().block(0, 0, j * blk_rows_, cols_);
        }
        auto operator()(int k) { return data_.template slice<2>(k).as_eigen_map(); }
    };

    template <typename DataLocs>
        requires(is_valid_data_locs_descriptor_v<DataLocs>)
    void eval_spatial_basis_at_(const DataLocs& locs) {
        if constexpr (std::is_same_v<DataLocs, matrix_t>) {   // pointwise sampling
            Psi_ = point_eval_(locs);
            D_ = vector_t::Ones(n_locs_).asDiagonal();
        } else {   // areal sampling
            const auto& [psi, measure_vect] = areal_eval_(locs);
            Psi_ = psi;
            D_ = measure_vect.asDiagonal();
        }
        fdapde_assert(n_locs_ == Psi_.rows());
        return;
    }
    template <typename GeoFrame> void eval_spatial_basis_at_(const GeoFrame& gf) {
        switch (gf.category(0)[0]) {
        case ltype::point: {
            const auto& spatial_index = geo_index_cast<0, POINT>(gf[0]);
            n_ = spatial_index.rows();
            if (spatial_index.points_at_dofs()) {
                Psi_.resize(n_, n_);
                Psi_.setIdentity();
            } else {
                Psi_ = point_eval_(spatial_index.coordinates());
            }
            D_ = vector_t::Ones(n_).asDiagonal();
            break;
        }
        case ltype::areal: {
            const auto& spatial_index = geo_index_cast<0, POLYGON>(gf[0]);
            n_ = spatial_index.rows();
            const auto& [psi, measure_vect] = areal_eval_(spatial_index.incidence_matrix());
            Psi_ = psi;
            D_ = measure_vect.asDiagonal();
            break;
        }
        }
        return;
    }
    // J(f,g) = \sum_{k=1}^m (y^k - \Psi*f^k)^T*(y^k - \Psi*f^k) + \lambda_S*(g^k)^T*(g^k) + \lambda_T*(l^k)^T*(l^k)
    double J_(const block_map_t& y, const block_map_t& x, double lambda_D, double lambda_T) const {
        double sse = 0;
        for (int t = 0; t < m_; ++t) {

	  // to be generalized wrt a generic weight matrix W_
	  
            sse += 1. / n_obs_ * (y(t) - Psi_ * x(0, t)).squaredNorm() + lambda_D * x(1, t).squaredNorm() +
                   lambda_T * x(2, t).squaredNorm();
        }
        return sse;
    }
   public:
    static constexpr int n_lambda = 2;
    using solver_category = ls_solver;

    fe_ls_separable_parallel() noexcept = default;
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_parallel(const std::string& formula, const GeoFrame& gf, Penalty&& penalty, const WeightMatrix& W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
	fdapde_assert(gf.n_layers() == 1);
        n_obs_  = gf[0].rows();
        n_locs_ = n_obs_;

        discretize(penalty);
        analyze_data(formula, gf, W);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_parallel(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) :
        fe_ls_separable_parallel(formula, gf, penalty, vector_t::Ones(gf[0].rows()).asDiagonal()) { }
    // construct with no data
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_parallel(const GeoFrame& gf, Penalty&& penalty, const WeightMatrix& W) : W_(W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
	fdapde_assert(gf.n_layers() == 1);
	n_obs_  = gf[0].rows();
	n_locs_ = n_obs_;

	// extract temporal mesh
        const auto& time_index = geo_index_cast<1, POINT>(gf[0]);
        const auto& time_coords = time_index.coordinates();
        m_ = time_coords.rows();
        fdapde_assert(m_ > 0 && time_coords.cols() == 1);
        DeltaT_ = time_coords(1, 0) - time_coords(0, 0);
        for (int i = 1; i < m_ - 1; ++i) {
            double lag_i = time_coords(i + 1, 0) - time_coords(i, 0);
            fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
        }

        discretize(penalty);
        u_.resize(n_dofs_ * m_);
        for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        eval_spatial_basis_at_(gf);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_parallel(const GeoFrame& gf, Penalty&& penalty) :
        fe_ls_separable_parallel(gf, penalty, vector_t::Ones(gf[0].rows()).asDiagonal()) { }

    template <typename Penalty> void discretize(Penalty&& penalty) {
        using BilinearForm = typename std::decay_t<Penalty>::BilinearForm;
        using LinearForm = typename std::decay_t<Penalty>::LinearForm;
        fdapde_static_assert(
          internals::is_valid_penalty_pair_v<BilinearForm FDAPDE_COMMA LinearForm>, INVALID_PENALTY_DESCRIPTION);
        using FeSpace = typename BilinearForm::TrialSpace;
        fdapde_static_assert(
          std::is_same_v<typename FeSpace::discretization_category FDAPDE_COMMA finite_element_tag>,
          NO_FINITE_ELEMENT_SPACE_DETECTED);
        // discretization
        const BilinearForm& bilinear_form = penalty.bilinear_form();
        const LinearForm& linear_form = penalty.linear_form();
        n_dofs_ = bilinear_form.n_dofs();
        internals::fe_mass_assembly_loop<FeSpace> mass_assembler(bilinear_form.trial_space());
        R0_ = mass_assembler.assemble();
        R1_ = bilinear_form.assemble();
	u_space_ = linear_form.assemble();
	// store handles for basis system evaluation at locations
        point_eval_ = [fe_space = bilinear_form.trial_space()](const matrix_t& locs) -> decltype(auto) {
            return internals::point_basis_eval(fe_space, locs);
        };
        areal_eval_ = [fe_space = bilinear_form.trial_space()](const binary_t& locs) -> decltype(auto) {
            return internals::areal_basis_eval(fe_space, locs);
        };
	// store numerical scheme parameters
	max_iter_ = penalty.max_iter();
	tol_ = penalty.tol();
	return;
    }
    // non-parametric fit
    // \sum_i w_i * (y_i - f(p_i))^2 + \int_D (Lf - u)^2
    template <typename DataLocs1, typename DataLocs2, typename WeightMatrix>
        requires(is_valid_data_locs_descriptor_v<DataLocs1> && is_valid_data_locs_descriptor_v<DataLocs2>)
    void analyze_data(const DataLocs1& locs1, const DataLocs2& locs2, const matrix_t& y, const WeightMatrix& W) {
        fdapde_static_assert(
          std::is_same_v<DataLocs2 FDAPDE_COMMA matrix_t>,
          ITERATIVE_SEPARABLE_PENALIZATION_REQUIRES_POINT_TEMPORAL_EVALUATIONS_ONLY);
        fdapde_assert(
          locs1.rows() > 0 && locs2.rows() > 0 && y.rows() == locs1.rows() * locs2.rows() && y.cols() == 1 &&
          W.rows() == locs1.rows() * locs2.rows() && W.rows() == W.cols());
        n_obs_ = y.rows();
        n_locs_ = n_obs_;
        n_covs_ = 0;
        eval_spatial_basis_at_(locs1);   // update \Psi matrix

        m_ = locs2.rows();
        fdapde_assert(m_ > 0 && locs2.cols() == 1);
        DeltaT_ = locs2(1, 0) - locs2(0, 0);
        for (int i = 1; i < m_ - 1; ++i) {
            double lag_i = locs2(i + 1, 0) - locs2(i, 0);
            fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
        }
	// update forcing
        if (u_.size() == 0) {
            u_.resize(n_dofs_ * m_);
            for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        }
        // update_response_and_weights(y, W);
        return;
    }
    // fit from formula
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix&) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_ONE_GEOFRAMES_ONLY);
        fdapde_assert(gf.n_layers() == 1 && gf[0].category()[1] == ltype::point);
        n_obs_ = gf[0].rows();
        n_locs_ = n_obs_;

        // extract time step and number of time instants
        const auto& time_index = geo_index_cast<1, POINT>(gf[0]);
        const auto& time_coords = time_index.coordinates();
        m_ = time_coords.rows();
        fdapde_assert(m_ > 0 && time_coords.cols() == 1);
        DeltaT_ = time_coords(1, 0) - time_coords(0, 0);
        for (int i = 1; i < m_ - 1; ++i) {
            double lag_i = time_coords(i + 1, 0) - time_coords(i, 0);
            fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
        }
        if (u_.size() == 0) {
            u_.resize(n_dofs_ * m_);
            for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        }
        eval_spatial_basis_at_(gf);   // update \Psi matrix
        // parse formula, extract response vector
        Formula formula_(formula);
        const auto& y_data = gf[0].data().template col<double>(formula_.lhs());
        y_.resize(n_locs_, y_data.blk_sz());
        y_data.assign_to(y_);

        // update_weights(vector_t::Ones(n_).asDiagonal());
    }

    // modifiers
    // void update_response(const vector_t& y) {
    //     fdapde_assert(Psi_.rows() > 0 && y.rows() == n_locs_ && y.cols() == 1);
    //     y_ = y;
    // 	b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * W_ * y;
    //     return;
    // }
    // template <typename WeightMatrix> void update_weights(const WeightMatrix& W) {
    //     fdapde_assert(Psi_.rows() > 0 && W.rows() == n_locs_ && W.rows() == W.cols());
    //     W_ = W;
    //     // W_ /= n_obs_; ---------------------- divide by n_obs_ or just by n_ ????
    //     b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * W_ * y_;
    //     W_changed_ = true;
    //     return;
    // }
    // template <typename WeightMatrix> void update_response_and_weights(const vector_t& y, const WeightMatrix& W) {
    //     fdapde_assert(
    //       Psi_.rows() > 0 && y.rows() == n_locs_ && y.cols() == 1 && W.rows() == W.cols() && W.rows() == n_);
    //     y_ = y;
    // 	update_weights(W);
    // 	return;
    // }
   private:
    // iterative scheme implementation
    template <typename ResponseT> auto fit_(ResponseT&& response, double lambda_D, double lambda_T) {
        std::array<double, n_lambda> lambda {lambda_D, lambda_T};
        // define auxiliary structures
        block_map_t y(response, n_);
        block_map_t u(u_, n_dofs_);
        matrix_t x_old_buff(3 * n_dofs_ * m_, response.cols()), x_new_buff(3 * n_dofs_ * m_, response.cols());
        block_map_t x_old(x_old_buff, 3 * n_dofs_, n_dofs_, response.cols());
        block_map_t x_new(x_new_buff, 3 * n_dofs_, n_dofs_, response.cols());
        double alpha = lambda_T / std::pow(DeltaT_, 2);

	const int n_threads = parallel_get_num_threads();
	
        {   // compute starting point (f^(k, 0), g^(k, 0), l^(k, 0)) k = 1 ... m
            if (lambda_saved_.value() != lambda) {
                SparseBlockMatrix<double, 2, 2> A_(
                  Psi_.transpose() * D_ * Psi_ / static_cast<double>(n_obs_), lambda_D * R1_.transpose(),
                  lambda_D * R1_, -lambda_D * R0_);
                invAs_.compute(A_);
            }

            std::vector<vector_t> tl_b(n_threads);
            for (auto& b : tl_b) { b.resize(2 * n_dofs_); }

            parallel_for(0, m_, [&, this] (int t) {
                tl_b[this_thread_id()] << Psi_.transpose() * D_ * y(t) / static_cast<double>(n_obs_), lambda_D * u(t);
                x_old.topRows(2, t) = invAs_.solve(tl_b[this_thread_id()]);
            });
	    
            x_old(2, 0).setZero();
            x_old(2, m_ - 1).setZero();
            for (int t = 1; t < m_ - 1; ++t) {
                x_old(2, t) = (x_old(0, t + 1) - 2 * x_old(0, t) + x_old(0, t - 1)) / std::pow(DeltaT_, 2);
            }
        }
        // iterative scheme initialization
        double Jold = std::numeric_limits<double>::max();
        double Jnew = J_(y, x_old, lambda_D, lambda_T);
        if (lambda_saved_.value() != lambda) {
            sparse_matrix_t Zero(n_dofs_, n_dofs_);
            SparseBlockMatrix<double, 3, 3> A_(
              Psi_.transpose() * D_ * Psi_ / static_cast<double>(n_obs_), lambda_D * R1_.transpose(), -2 * alpha * R0_,
              lambda_D * R1_, -lambda_D * R0_, Zero, -2 * alpha * R0_, Zero, -lambda_T * R0_);
            invA_.compute(A_);
        }
        std::vector<vector_t> tl_b(n_threads);
        for (auto& b : tl_b) { b.resize(3 * n_dofs_); }
        // iterative loop
        x_new(2, 0).setZero();
        x_new(2, m_ - 1).setZero();
        int i = 1;
        while (i < max_iter_ && std::abs((Jnew - Jold) / Jnew) > tol_) {
            TaskGraph g;
            g.add_node([&, this] {
                // at step 0: f^(-1, i-1) = l^(-1, i-1) = 0
                tl_b[this_thread_id()] << Psi_.transpose() * D_ * y(0) / static_cast<double>(n_obs_) -
                                            alpha * R0_ * x_old(2, 1),
                  lambda_D * u(0), -alpha * R0_ * x_old(0, 1);
                x_new.topRows(2, 0) = invA_.solve(tl_b[this_thread_id()]).topRows(2 * n_dofs_);   // l^(0) = 0
            });
            // general step
            g.add_node([&, this] {
                parallel_for(1, m_ - 1, [&, this](int t) {
                    tl_b[this_thread_id()] << Psi_.transpose() * D_ * y(t) / static_cast<double>(n_obs_) -
                                                alpha * R0_ * (x_old(2, t + 1) + x_old(2, t - 1)),
                      lambda_D * u(t), -alpha * R0_ * (x_old(0, t + 1) + x_old(0, t - 1));
                    x_new(t) = invA_.solve(tl_b[this_thread_id()]);
                });
            });
            g.add_node([&, this] {
                // at step m_ - 1: f^(m+1, i-1) = l^(m+1, i-1) = 0
                tl_b[this_thread_id()] << Psi_.transpose() * D_ * y(m_ - 1) / static_cast<double>(n_obs_) -
                                            alpha * R0_ * x_old(2, m_ - 1),
                  lambda_D * u(m_ - 1), -alpha * R0_ * x_old(0, m_ - 1);
                x_new.topRows(2, m_ - 1) = invA_.solve(tl_b[this_thread_id()]).topRows(2 * n_dofs_);   // l^(m_ - 1) = 0
            });
	    parallel_execute(g);
	    
            // prepare for next iteration
            Jold = Jnew;
            Jnew = J_(y, x_new, lambda_D, lambda_T);

            x_old = x_new;
            i++;
        }
        // return result
        vector_t f(n_dofs_ * m_, response.cols());
        vector_t g(n_dofs_ * m_, response.cols());
        for (int i = 0; i < m_; ++i) {
            f.middleRows(i * n_dofs_, n_dofs_) = x_old(0, i);
            g.middleRows(i * n_dofs_, n_dofs_) = x_old(1, i);
        }
        lambda_saved_ = std::array<double, n_lambda> {lambda_D, lambda_T};
        return std::make_pair(f, g);
    }
   public:
    // main fit entry point
    const vector_t& fit(double lambda_D, double lambda_T) {
        const auto& [f, g] = fit_(y_, lambda_D, lambda_T);
        f_ = f;
        g_ = g;
        return f_;
    }
    template <typename LambdaT>
        requires(internals::is_vector_like_v<LambdaT>)
    const vector_t& fit(LambdaT&& lambda) {
        fdapde_assert(lambda.size() == n_lambda);
        return fit(lambda[0], lambda[1]);
    }

    // hutchinson approximation for Tr[S]
    double edf(int r = 100, int seed = random_seed) {
        std::cout << "HUTCH NORMALE" << std::endl;
        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::duration;
        using std::chrono::seconds;

        auto t1 = high_resolution_clock::now();

        fdapde_assert(lambda_saved_.has_value());
        if (!Us_.has_value() || r != Us_->rows()) {   // force reconstruction if r differs from old
            int seed_ = (seed == random_seed) ? std::random_device()() : seed;

            std::default_random_engine generator(seed);
            std::bernoulli_distribution distribution(0.5);

// con queto no, anche se l'andamento del GCV è uguale, da capire perchè
            // std::mt19937 rng(seed_);
            // rademacher_distribution rademacher;
            Us_ = matrix_t::Zero(n_locs_, r);
            for (int i = 0; i < n_locs_; ++i) {
                for (int j = 0; j < r; ++j) { 
                    //Us_->operator()(i, j) = rademacher(rng); 
                    Us_->operator()(i, j) = distribution(generator) ? 1.0 : -1.0;
                }
            }

// codice di prima
            //std::mt19937 rng(seed_);
            //rademacher_distribution rademacher;
            //Us_->resize(n_locs_, r);
            //for (int i = 0; i < n_locs_; ++i) {
            //    for (int j = 0; j < r; ++j) { Us_->operator()(i, j) = rademacher(rng); }
            //}
        }
        // Tr[S] \approx \sum_{i=1}^r (u_i^\top * S * u_i)
        double trS = 0;
        for (int i = 0; i < r; ++i) {
            const auto& [f, g] = fit_(Us_->col(i), lambda_saved_->operator[](0), lambda_saved_->operator[](1));
            vector_t fn(n_ * m_);
            for (int i = 0; i < m_; ++i) { fn.middleRows(i * n_, n_) = Psi_ * f.middleRows(i * n_dofs_, n_dofs_); }
            trS += Us_->col(i).dot(fn);
        }

        auto t2 = high_resolution_clock::now();
        auto ms_int1 = duration_cast<milliseconds>(t2 - t1);
        std::cout << "ms_int1: " << ms_int1.count() << "s" << std::endl;
        std::cout << "EDF: " << (trS / r) << std::endl;

        return trS / r;
    }
    template <typename... LambdaT>
        requires(
          (sizeof...(LambdaT) == 1 && (internals::is_vector_like_v<LambdaT> && ...)) ||
          (sizeof...(LambdaT) == n_lambda && (std::is_floating_point_v<LambdaT> && ...)))
    double edf(const LambdaT&... lambda, int r = 100, int seed = random_seed) {
        std::array<double, n_lambda> lambda_;
        if constexpr (sizeof...(LambdaT) == 1) {
            internals::for_each_index_and_args<sizeof...(LambdaT)>([&]<int Ns_, typename Ts_>(const Ts_& ts) {
                fdapde_assert(ts.size() == n_lambda && ts[0] > 0 && ts[1] > 0);
                lambda_[0] = ts[0];
                lambda_[1] = ts[1];
            });
        } else {
            std::array<double, n_lambda> lambda__ {static_cast<double>(lambda)...};
            fdapde_assert(lambda__[0] > 0 && lambda__[1] > 0);
	    lambda_ = lambda__;
        }
        return edf(r, seed);
    }
    vector_t fn() const {
        vector_t fn_(n_ * m_);
        for (int i = 0; i < m_; ++i) { fn_.middleRows(i * n_, n_) = Psi_ * f_.middleRows(i * n_dofs_, n_dofs_); }
        return fn_;
    }

    // observers
    int n_dofs() const { return n_dofs_; }
    const sparse_matrix_t& mass() const { return R0_; }
    const sparse_matrix_t& stiff() const { return R1_; }
    const sparse_matrix_t& Psi() const { return Psi_; }
    const vector_t& force() const { return u_; }
    const vector_t& f() const { return f_; }
    const vector_t& beta() const { return beta_; }
    const vector_t& misfit() const { return g_; }
    const vector_t& response() const { return y_; }
   protected:
    std::optional<std::array<double, n_lambda>> lambda_saved_ = std::array<double, n_lambda> {-1, -1};
    sparse_solver_t invA_, invAs_;
    // matrices for hutchinson stochastic estimation of Tr[S]
    std::optional<matrix_t> Us_;

    int n_dofs_ = 0, n_obs_ = 0, n_covs_ = 0;
    int n_locs_ = 0, n_ = 0, m_ = 0;   // n_: number of spatial locations, m_: number of time instants

    sparse_matrix_t R0_;    // n_dofs x n_dofs matrix [R0]_{ij} = \int_D \psi_i * \psi_j
    sparse_matrix_t R1_;    // n_dofs x n_dofs matrix [R1]_{ij} = \int_D a(\psi_i, \psi_j)
    sparse_matrix_t Psi_;   // n_obs x n_dofs matrix [Psi]_{ij} = \psi_j(p_i)
    vector_t u_;            // (n_dofs * m) x 1 vector u = [u_1 + + R0_*s / DeltaT, u_2, \ldots, u_n]
    vector_t u_space_;
    diag_matrix_t D_;       // vector of regions' measures (areal sampling)
    mutable sparse_solver_t invR0_;
    vector_t f_, beta_, g_;
    // basis system evaluation handles
    std::function<sparse_matrix_t(const matrix_t& locs)> point_eval_;
    std::function<std::pair<sparse_matrix_t, vector_t>(const binary_t& locs)> areal_eval_;

    vector_t y_;          // n_obs x 1 observation vector
    sparse_matrix_t W_;   // n_obs x n_obs matrix of observation weights
    bool W_changed_;

    int max_iter_;   // maximum number of iterations
    double tol_;     // convergence tolerance
    double DeltaT_;
};

}   // namespace internals

template <typename Penalty>
    requires(internals::is_pair_v<Penalty>)
struct fe_ls_separable_parallel {
    using solver_t = internals::fe_ls_separable_parallel;
   private:
    struct penalty_packet {
        using BilinearForm = std::tuple_element_t<0, std::decay_t<Penalty>>;
        using LinearForm = std::tuple_element_t<1, std::decay_t<Penalty>>;
       private:
        BilinearForm bilinear_form_;
        LinearForm linear_form_;
        int max_iter_ = 50;
        double tol_ = 1e-4;
       public:
        penalty_packet(const BilinearForm& bilinear_form, const LinearForm& linear_form, int max_iter, double tol) :
            bilinear_form_(bilinear_form), linear_form_(linear_form), max_iter_(max_iter), tol_(tol) { }
        // observers
        const BilinearForm& bilinear_form() const { return bilinear_form_; }
        const LinearForm& linear_form() const { return linear_form_; }
        int max_iter() const { return max_iter_; }
        double tol() const { return tol_; }
    };
   public:
    fe_ls_separable_parallel(const Penalty& penalty, int max_iter = 50, double tol = 1e-4) :
        penalty_(std::get<0>(penalty), std::get<1>(penalty), max_iter, tol) { }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};

}   // namespace fdapde

#endif // __FE_LS_SEPARABLE_SOLVER_H__
