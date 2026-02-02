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

#include<fdaPDE/optimization.h>
#include<random>

int main(int argc, char** argv){
    double lower = -5;
    double upper = 5;
    std::uniform_real_distribution<double> unif(lower,upper);
    std::default_random_engine re;

    int grid_size = std::stoi(argv[1]);


    // funzione x^2 + y^2
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return std::pow(p[0], 2) + std::pow(p[1], 2); })> objective;

    //rastrigin function
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 20 + p[0]*p[0] + p[1]*p[1] - 10*std::cos(2*M_PI*p[0]) - 10*std::cos(2*M_PI*p[1]); })> rastrigin;

    // matrix
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) {
        int n = 50;
        Eigen::MatrixXd A = Eigen::MatrixXd::Constant(n, n, 2.0);
        Eigen::MatrixXd B = Eigen::MatrixXd::Constant(n, n, 3.0);
        Eigen::MatrixXd C = A * B; // moltiplicazione costosa

        return C.sum() + p[0]*p[0] + p[1]*p[1];
    })> matrix_function;

    // definizione di griglia di possibili valori
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> grid;
    grid.resize(grid_size,2);

    // grid da popolare con la griglia dei valori da esplorare

    for(int i =0; i<grid.rows();++i){
        grid(i,0) = unif(re);
        grid(i,1) = unif(re);
    }

    // definizione dell'ottimizzatore 
    fdapde::GridSearch<2> opt;

    auto start2 = std::chrono::high_resolution_clock::now();

    opt.optimize(rastrigin, grid); // <- da modificare questo step

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    std::cout<<duration2.count()<<",";
    
    return 0;
}
