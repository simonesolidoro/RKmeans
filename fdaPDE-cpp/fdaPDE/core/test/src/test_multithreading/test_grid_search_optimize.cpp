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
    double lower =-5;
    double upper = 5;
    std::uniform_real_distribution<double> unif(lower,upper);
    std::default_random_engine re;
    
    int grid_size = 0;
    int n_threads = 1;
    int granularity = -1; 
    std::cout<<"numero elementi in griglia: ";
    std::cin>>grid_size;
    // std::cout<<"numero worker in threadpool: ";
    // std::cin>>n_threads;
    // std::cout<<"granularity: ";
    // std::cin>>granularity;
    std::cout<<std::endl;

    //rastrigin function min in (0,0)
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 20 + p[0]*p[0] + p[1]*p[1] - 10*std::cos(2*M_PI*p[0]) - 10*std::cos(2*M_PI*p[1]); })> rastrigin;
    //rosenbrock. min in (1,1)
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return std::pow(1 - p[0], 2) + 100 * std::pow(p[1] - p[0]*p[0], 2); })> rosenbrock;

    // definizione di griglia di possibili valori
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> grid;
    grid.resize(grid_size,2);

    // grid da popolare con la griglia dei valori da esplorare
    for(int i =0; i<grid.rows();++i){
        grid(i,0) = unif(re);
        grid(i,1) = unif(re);
    }
    
 { //rastrigin
    std::cout<<"=========minimization of Rastrigin function ==================================="<<std::endl;
    // //creazione threadpool per versioni con Tp in input
    // fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::max_load_stealing> Tp(1024, n_threads);


    //---------------------- no parallel---------------------- ---------------------- ---------------------- ---------------------- 
    std::cout<<"========================sequenziale========================"<<std::endl;
    fdapde::GridSearch<2> opt2;

    auto start3 = std::chrono::high_resolution_clock::now();

    opt2.optimize(rastrigin, grid); // <- da modificare questo step

    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    std::cout<<"tempo impiegato: "<<duration3.count()<<","<<std::endl;
    std::cout<<"value: "<<opt2.value()<<std::endl; 
    std::cout<<"optimum: "<<opt2.optimum()<<std::endl; 
    std::cout<<std::endl;
    

//---------------------- parallel: optimize (parallel_for)---------------------- ---------------------- ---------------------- ---------------------- 
    std::cout<<"========================parallel_for optimize, granularity:"<<granularity<<", threads: "<<n_threads<<"========================"<<std::endl;
    // definizione dell'ottimizzatore 
    fdapde::GridSearch<2> opt3;

    auto start4 = std::chrono::high_resolution_clock::now();

    //opt3.optimize(rastrigin, grid, execution::par,Tp,granularity);  
    opt3.optimize(fdapde::execution_par,rastrigin, grid);  

    auto end4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4);  
    std::cout<<"tempo impiegato: "<<duration4.count()<<","<<std::endl;
    std::cout<<"value: "<<opt3.value()<<std::endl; 
    std::cout<<"optimum: "<<opt3.optimum()<<std::endl;
    std::cout<<std::endl;

    // //prima scommentare in grid_search.h il metodo optimize_variadic
    // //---------------------- parallel_variadic : optimize (parallel_for_granularity_variadic)---------------------- ---------------------- ---------------------- ---------------------- 
    // std::cout<<"========================parallel_for_granularity_variadic, gran:"<<granularity<<", threads: "<<n_threads<<"========================"<<std::endl;
    // // definizione dell'ottimizzatore 
    // fdapde::GridSearch<2> opt4;

    // auto start5 = std::chrono::high_resolution_clock::now();

    // opt4.optimize_variadic(rastrigin, grid, execution::par,Tp,granularity);  

    // auto end5 = std::chrono::high_resolution_clock::now();
    // auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);  
    // std::cout<<"tempo impiegato: "<<duration5.count()<<","<<std::endl;
    // std::cout<<"value: "<<opt4.value()<<std::endl; 
    // std::cout<<"optimum: "<<opt4.optimum()<<std::endl;
    // std::cout<<std::endl;

 }

    return 0;
}
