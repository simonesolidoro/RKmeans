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

#ifndef __FE_LS_SEPARABLE_SOLVER_FD_H__
#define __FE_LS_SEPARABLE_SOLVER_FD_H__

namespace fdapde {
namespace internals {

// finite difference monolithic solver
class fe_ls_separable_fd {
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

    template <typename DataLocs>
        requires(is_valid_data_locs_descriptor_v<DataLocs>)
    void eval_basis_at_(const DataLocs& locs) {
        if constexpr (std::is_same_v<DataLocs, matrix_t>) {   // pointwise sampling
            Psi__ = point_eval_(locs);
            D_ = vector_t::Ones(n_locs_).asDiagonal();
        } else {   // areal sampling
            const auto& [psi, measure_vect] = areal_eval_(locs);
            Psi__ = psi;
            D_ = measure_vect.asDiagonal();
        }
        fdapde_assert(n_locs_ == Psi__.rows());
        return;
    }
    template <typename GeoFrame> void eval_basis_at_(const GeoFrame& gf) {
        switch (gf.category(0)[0]) {
        case ltype::point: {
            const auto& spatial_index = geo_index_cast<0, POINT>(gf[0]);
            n_ = spatial_index.rows();
            if (spatial_index.points_at_dofs()) {
                Psi__.resize(n_, n_);
                Psi__.setIdentity();
            } else {
                Psi__ = point_eval_(spatial_index.coordinates());
            }
            D_ = vector_t::Ones(n_).asDiagonal();
            break;
        }
        case ltype::areal: {
            const auto& spatial_index = geo_index_cast<0, POLYGON>(gf[0]);
            n_ = spatial_index.rows();
            const auto& [psi, measure_vect] = areal_eval_(spatial_index.incidence_matrix());
            Psi__ = psi;
            D_ = measure_vect.asDiagonal();
            break;
        }
        }
        return;
    }
   public:
    static constexpr int n_lambda = 2;
    using solver_category = ls_solver;

    fe_ls_separable_fd() noexcept = default;
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_fd(const std::string& formula, const GeoFrame& gf, Penalty&& penalty, const WeightMatrix& W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
        fdapde_assert(gf.n_layers() == 1);
        n_obs_ = gf[0].rows();
        n_locs_ = n_obs_;

        discretize(penalty);
        analyze_data(formula, gf, W);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_fd(const std::string& formula, const GeoFrame& gf, Penalty&& penalty) :
        fe_ls_separable_fd(formula, gf, penalty, vector_t::Ones(gf[0].rows()).asDiagonal()) { }
    // construct with no data
    template <typename GeoFrame, typename Penalty, typename WeightMatrix>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_fd(const GeoFrame& gf, Penalty&& penalty, const WeightMatrix& W) : W_(W) {
        fdapde_static_assert(GeoFrame::Order == 2, THIS_CLASS_IS_FOR_ORDER_TWO_GEOFRAMES_ONLY);
        fdapde_assert(gf.n_layers() == 1);
        n_obs_ = gf[0].rows();
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
        eval_basis_at_(gf);
        tensorize_(m_);
    }
    template <typename GeoFrame, typename Penalty>
        requires(is_valid_penalty_v<Penalty>)
    fe_ls_separable_fd(const GeoFrame& gf, Penalty&& penalty) :
        fe_ls_separable_fd(gf, penalty, vector_t::Ones(gf[0].rows()).asDiagonal()) { }

    void tensorize_(int m) {
        sparse_matrix_t Im(m, m);   // m x m identity matrix
        Im.setIdentity();
        R0_ = kronecker(Im, R0__);
        R1_ = kronecker(Im, R1__);

        {
            // assemble matrix associated with central finite differences in time D_
            std::vector<Eigen::Triplet<double>> triplet_list;
            triplet_list.reserve(3 * (m - 2) + 4);
            // start assembly loop
            triplet_list.emplace_back(0, 0, -2.0 / (DeltaT_ * DeltaT_));
            triplet_list.emplace_back(0, 1,  1.0 / (DeltaT_ * DeltaT_));
            for (int i = 1; i < m - 1; ++i) {
                triplet_list.emplace_back(i, i - 1, 1.0 / (DeltaT_ * DeltaT_));
                triplet_list.emplace_back(i, i,    -2.0 / (DeltaT_ * DeltaT_));
                triplet_list.emplace_back(i, i + 1, 1.0 / (DeltaT_ * DeltaT_));
            }
            triplet_list.emplace_back(m - 1, m - 2,  1.0 / (DeltaT_ * DeltaT_));
            triplet_list.emplace_back(m - 1, m - 1, -2.0 / (DeltaT_ * DeltaT_));

            L__.resize(m, m);
            L__.setFromTriplets(triplet_list.begin(), triplet_list.end());
            L__.makeCompressed();
            L_ = kronecker(L__, R0__);
        }

        u_.resize(n_dofs_ * m);
        for (int i = 0; i < m; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u__; }
        Psi_ = kronecker(Im, Psi__);
    }

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
        R0__ = mass_assembler.assemble();
        R1__ = bilinear_form.assemble();
        u__ = linear_form.assemble();
        // store handles for basis system evaluation at locations
        point_eval_ = [fe_space = bilinear_form.trial_space()](const matrix_t& locs) -> decltype(auto) {
            return internals::point_basis_eval(fe_space, locs);
        };
        areal_eval_ = [fe_space = bilinear_form.trial_space()](const binary_t& locs) -> decltype(auto) {
            return internals::areal_basis_eval(fe_space, locs);
        };
        return;
    }
    // non-parametric fit
    // \sum_i w_i * (y_i - f(p_i))^2 + \int_D (Lf - u)^2
    // template <typename DataLocs1, typename DataLocs2, typename WeightMatrix>
    //     requires(is_valid_data_locs_descriptor_v<DataLocs1> && is_valid_data_locs_descriptor_v<DataLocs2>)
    // void analyze_data(const DataLocs1& locs1, const DataLocs2& locs2, const matrix_t& y, const WeightMatrix& W) {
    //     fdapde_static_assert(
    //       std::is_same_v<DataLocs2 FDAPDE_COMMA matrix_t>,
    //       ITERATIVE_SEPARABLE_PENALIZATION_REQUIRES_POINT_TEMPORAL_EVALUATIONS_ONLY);
    //     fdapde_assert(
    //       locs1.rows() > 0 && locs2.rows() > 0 && y.rows() == locs1.rows() * locs2.rows() && y.cols() == 1 &&
    //       W.rows() == locs1.rows() * locs2.rows() && W.rows() == W.cols());
    //     n_obs_ = y.rows();
    //     n_locs_ = n_obs_;
    //     n_covs_ = 0;
    //     eval_basis_at_(locs1);   // update \Psi matrix

    //     int m = locs2.rows();
    //     fdapde_assert(m > 0 && locs2.cols() == 1);
    //     DeltaT_ = locs2(1, 0) - locs2(0, 0);
    //     for (int i = 1; i < m - 1; ++i) {
    //         double lag_i = locs2(i + 1, 0) - locs2(i, 0);
    //         fdapde_assert(DeltaT_ > 0 && lag_i > 0 && almost_equal(DeltaT_ FDAPDE_COMMA lag_i));
    //     }
    //     if (m != m_) {   // re-tensorize all if number of time steps chanded
    //         m_ = m;
    //         tensorize_(m_);
    //     } else {
    //         sparse_matrix_t Im(m, m);   // m x m identity matrix
    //         Im.setIdentity();
    //         Psi_ = kronecker(Im, Psi__);
    //     }
    //     y_ = y;
    //     b_.resize(3 * m_ * n_dofs_, y.cols());

    //     // update forcing
    //     // if (u_.size() == 0) {
    //     //     u_.resize(n_dofs_ * m_);
    //     //     for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
    //     // }
    //     // update_response_and_weights(y, W);
    //     return;
    // }
    // fit from formula
    template <typename GeoFrame, typename WeightMatrix>
    void analyze_data(const std::string& formula, const GeoFrame& gf, const WeightMatrix& W) {
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
        // if (u_.size() == 0) {
        //     u_.resize(n_dofs_ * m_);
        //     for (int i = 0; i < m_; ++i) { u_.segment(i * n_dofs_, n_dofs_) = u_space_; }
        // }
        eval_basis_at_(gf);   // update \Psi matrix
        tensorize_(m_);       // magari da ottimizzare

        // parse formula, extract response vector
        Formula formula_(formula);
        const auto& y_data = gf[0].data().template col<double>(formula_.lhs());
        y_.resize(n_locs_, y_data.blk_sz());
        y_data.assign_to(y_);

        b_.resize(3 * m_ * n_dofs_, y_.cols());
        b_.setZero();
        // b_.block(0, 0, m_ * n_dofs_, 1) = Psi_.transpose() * y_; ------------------------------ da fare quando conosciamo W

        update_response_and_weights(y_, W);
    }

    // modifiers
    void update_response(const vector_t& y) {
        fdapde_assert(Psi_.rows() > 0 && y.rows() == Psi_.rows() && y.cols() == 1);
        y_ = y;

	// here correction for missing observations
	
        b_.block(0, 0, m_ * n_dofs_, 1) = Psi_.transpose() * W_ * y;
        return;
    }
    template <typename WeightMatrix> void update_weights(const WeightMatrix& W) {
        fdapde_assert(Psi_.rows() > 0 && W.rows() == n_locs_ && W.rows() == W.cols());
        W_ = W;
        W_ /= n_obs_;
        b_.block(0, 0, m_ * n_dofs_, 1) = Psi_.transpose() * W_ * y_;
        W_changed_ = true;
        return;
    }
    template <typename WeightMatrix> void update_response_and_weights(const vector_t& y, const WeightMatrix& W) {
        fdapde_assert(
          Psi_.rows() > 0 && y.rows() == n_locs_ && y.cols() == 1 && W.rows() == W.cols() && W.rows() == n_locs_);
        y_ = y;   // here we update twice y_, if called by analyze_data

        // correction for missing observations

        update_weights(W);   // here we assume that W is provided unnormalized
        return;
    }
   private:
    // iterative scheme implementation
    template <typename ResponseT> auto fit_(ResponseT&& response, double lambda_D, double lambda_T) {
	
    }
   public:
    // main fit entry point
    const vector_t& fit(double lambda_D, double lambda_T) {
        std::array<double, n_lambda> lambda {lambda_D, lambda_T};
        if (lambda_saved_.value() != lambda || W_changed_) {
            // assemble system matrix for the nonparameteric part

	  // qui manca supporto per missing values (PsiNA), areal data (matrice D_), weights W_
	 

            SparseBlockMatrix<double, 3, 3> A(
              Psi_.transpose() * W_ * Psi_ , lambda_D * R1_.transpose(), lambda_T * L_  ,
	      lambda_D * R1_               , -lambda_D * R0_           , 0              ,
              lambda_T * L_                , 0                         , -lambda_T * R0_);
            invA_.compute(A);
            W_changed_ = false;
        }
        if (lambda_saved_.value() != lambda) {
            // linear system rhs
            b_.block(n_dofs_ * m_, 0, n_dofs_ * m_, 1) = lambda_D * u_;
        }
        lambda_saved_ = lambda;
        vector_t x;
        if (n_covs_ == 0) {   // nonparametric case
            x = invA_.solve(b_);
            f_ = x.head(n_dofs_ * m_);
        } else {   // parametric case
            // x = woodbury_system_solve(invA_, U_, XtWX_, V_, b_);
            // f_ = x.head(n_dofs_ * m_);
            // beta_ = invXtWXXtW_ * (y_ - Psi_ * f_);
        }
        g_ = x.block(n_dofs_ * m_, 0, n_dofs_ * m_, 1);
        return f_;
    }
    template <typename LambdaT>
        requires(internals::is_vector_like_v<LambdaT>)
    const vector_t& fit(LambdaT&& lambda) {
        fdapde_assert(lambda.size() == n_lambda);
        return fit(lambda[0], lambda[1]);
    }

    // exact edf
    double edf_exact() {
        // build penalization matrix
        Eigen::SparseLU<Eigen::SparseMatrix<double>> invR0(R0_);
        double lambda_D = lambda_saved_.value()[0];
        double lambda_T = lambda_saved_.value()[1];
        Eigen::Matrix<double, Dynamic, Dynamic> P =
          lambda_D * (R1_.transpose() * invR0.solve(R1_)) + lambda_T * (L_ * invR0.solve(L_));

	// E = Psi^\top * Psi + P
        Eigen::Matrix<double, Dynamic, Dynamic> E = Psi_.transpose() * W_ * Psi_ + P;
        Eigen::PartialPivLU<Eigen::Matrix<double, Dynamic, Dynamic>> invE(E);

	// S = Psi * E^{-1} * Psi^\top
        Eigen::Matrix<double, Dynamic, Dynamic> Psi_dense = W_ * Psi_;
        Eigen::Matrix<double, Dynamic, Dynamic> S = Psi_dense * invE.solve(Psi_dense.transpose());

        return S.trace();
    }

    // hutchinson approximation for Tr[S]
    double edf(int r = 100, int seed = random_seed) {
        fdapde_assert(lambda_saved_.has_value());
        if (!Ys_.has_value() || !Bs_.has_value()) {
            int seed_ = (seed == random_seed) ? std::random_device()() : seed;


            //std::mt19937 rng(seed_);
            //rademacher_distribution rademacher;
            //Us_->resize(n_locs_, r);
            //for (int i = 0; i < n_locs_; ++i) {
            //    for (int j = 0; j < r; ++j) { Us_->operator()(i, j) = rademacher(rng); }
            //}

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

            Ys_ = Us_->transpose() * Psi_;
            Bs_ = matrix_t::Zero(3 * m_ * n_dofs_, r);   // implicitly enforce homogeneous forcing
        }
        // if (n_covs_ == 0) {
        Bs_->topRows(m_ * n_dofs_) = Psi_.transpose() * W_ * (*Us_);
        // } else {
        //     Bs_->topRows(m_ * n_dofs_) = -Psi_.transpose() * D_ * internals::lmbQ(W_, X_, invXtWX_, *Us_);
        // }
        matrix_t x = invA_.solve(*Bs_);
        double trS = 0;   // monte carlo Tr[S] approximation
        for (int i = 0; i < r; ++i) { trS += Ys_->row(i).dot(x.col(i).head(m_ * n_dofs_)); }
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
    // vector_t fn() const {
    //     vector_t fn_(n_ * m_);
    //     for (int i = 0; i < m_; ++i) { fn_.middleRows(i * n_, n_) = Psi_ * f_.middleRows(i * n_dofs_, n_dofs_); }
    //     return fn_;
    // }
  
    vector_t fn() const { return Psi_ * f_; }
  
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
    sparse_solver_t invA_;
    matrix_t b_;
    // matrices for hutchinson stochastic estimation of Tr[S]
    std::optional<matrix_t> Ys_;
    std::optional<matrix_t> Bs_;
    std::optional<matrix_t> Us_;


    int n_dofs_ = 0, n_obs_ = 0, n_covs_ = 0;
    int n_locs_ = 0, n_ = 0, m_ = 0;   // n_: number of spatial locations, m_: number of time instants

    // not tensorized quantities
    sparse_matrix_t R0__;    // n_dofs x n_dofs matrix [R0]_{ij} = \int_D \psi_i * \psi_j
    sparse_matrix_t R1__;    // n_dofs x n_dofs matrix [R1]_{ij} = \int_D a(\psi_i, \psi_j)
    vector_t u__;            // n_dofs x 1 vector [u]_i = \int_D u * \psi_i
    sparse_matrix_t Psi__;   // n_locs x n_dofs matrix [Psi]_{ij} = \psi_j(p_i)
    sparse_matrix_t L__;

    sparse_matrix_t R0_;    // n_dofs x n_dofs matrix [R0]_{ij} = \int_D \psi_i * \psi_j
    sparse_matrix_t R1_;    // n_dofs x n_dofs matrix [R1]_{ij} = \int_D a(\psi_i, \psi_j)
    sparse_matrix_t Psi_;   // n_obs x n_dofs matrix [Psi]_{ij} = \psi_j(p_i)
    sparse_matrix_t L_;
    vector_t u_;            // (n_dofs * m) x 1 vector u = [u_1 + + R0_*s / DeltaT, u_2, \ldots, u_n]
    diag_matrix_t D_;       // vector of regions' measures (areal sampling)
  
    mutable sparse_solver_t invR0_;
    vector_t f_, beta_, g_;
    // basis system evaluation handles
    std::function<sparse_matrix_t(const matrix_t& locs)> point_eval_;
    std::function<std::pair<sparse_matrix_t, vector_t>(const binary_t& locs)> areal_eval_;

    vector_t y_;          // n_obs x 1 observation vector
    sparse_matrix_t W_;   // n_obs x n_obs matrix of observation weights
    bool W_changed_;

    double DeltaT_;
};

}   // namespace internals


template <typename Penalty>
    requires(internals::is_pair_v<Penalty>)
struct fe_ls_separable_fd {
    using solver_t = internals::fe_ls_separable_fd;
   private:
    struct penalty_packet {
        using BilinearForm = std::tuple_element_t<0, std::decay_t<Penalty>>;
        using LinearForm = std::tuple_element_t<1, std::decay_t<Penalty>>;
       private:
        BilinearForm bilinear_form_;
        LinearForm linear_form_;
       public:
        penalty_packet(const BilinearForm& bilinear_form, const LinearForm& linear_form) :
            bilinear_form_(bilinear_form), linear_form_(linear_form) { }
        // observers
        const BilinearForm& bilinear_form() const { return bilinear_form_; }
        const LinearForm& linear_form() const { return linear_form_; }
    };
   public:
    fe_ls_separable_fd(const Penalty& penalty) : penalty_(std::get<0>(penalty), std::get<1>(penalty)) { }
    const penalty_packet& get() const { return penalty_; }
   private:
    penalty_packet penalty_;
};
  
}   // namespace fdapde

#endif   // __FE_LS_SEPARABLE_SOLVER_FD_H__
