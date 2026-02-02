#include<fdaPDE/multithreading.h>
#include<fdaPDE/optimization.h>
#include <fdaPDE/finite_elements.h>
int main(int argc, char** argv){
    int n_thread= std::stoi(argv[1]);

//sequenziale vs parallelo garn1 vs parallelo gran_input
{//==================== GRID SEARCH ==============================================
    // default granularity
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> grid_sizes = {12800000}; //
    std::vector<int> runs_vett = {10};//
    std::vector<std::vector<std::chrono::microseconds>> tempi_seq(grid_sizes.size()); // vect esterno un elemento per ogni grid_size, quelli interni sono tempi di runs
    std::vector<std::vector<std::chrono::microseconds>> tempi_par(grid_sizes.size());
    std::vector<std::vector<std::chrono::microseconds>> tempi_par_graninput(grid_sizes.size());
    std::vector<std::vector<std::chrono::microseconds>> tempi_par_reduce(grid_sizes.size());
    std::vector<bool> samesame;
    double lower = -5;
    double upper = 5;
    std::uniform_real_distribution<double> unif(lower,upper);
    std::default_random_engine re;    
    //rastrigin function
    fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 20 + p[0]*p[0] + p[1]*p[1] - 10*std::cos(2*M_PI*p[0]) - 10*std::cos(2*M_PI*p[1]); })> rastrigin;
    //loop per diverse size di griglia in grid_sizes
    for(int gs = 0; gs< grid_sizes.size(); gs++){
        //crea griglia
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> grid;
        grid.resize(grid_sizes[gs],2);
        for(int i =0; i<grid.rows();++i){
            grid(i,0) = unif(re);
            grid(i,1) = unif(re);
        } 
    {//seq
        if(n_thread == 2){
            for (int run = 0; run<runs_vett[gs]; run ++){
                fdapde::GridSearch<2> opt;
                start = std::chrono::high_resolution_clock::now();
                opt.optimize(rastrigin, grid);
                end = std::chrono::high_resolution_clock::now();
                tempi_seq[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            }
        }
    }
    {//parallel
        for (int run = 0; run<runs_vett[gs]; run ++){
            fdapde::threadpool<fdapde::steal::random> Tp(1024, n_thread);
            fdapde::GridSearch<2> opt;
            start = std::chrono::high_resolution_clock::now();
            opt.optimize(rastrigin, grid, execution::par,Tp,-1);
            end = std::chrono::high_resolution_clock::now();
            tempi_par[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
        }
    }
    {//parallel
        for (int run = 0; run<runs_vett[gs]; run ++){
            fdapde::threadpool<fdapde::steal::random> Tp(1024, n_thread);
            fdapde::GridSearch<2> opt;
            start = std::chrono::high_resolution_clock::now();
            opt.optimize_variadic(rastrigin, grid, execution::par,Tp,-1);
            end = std::chrono::high_resolution_clock::now();
            tempi_par_graninput[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
        } 
    }
        {//parallel
        for (int run = 0; run<runs_vett[gs]; run ++){
            fdapde::threadpool<fdapde::steal::random> Tp(1024, n_thread);
            fdapde::GridSearch<2> opt;
            start = std::chrono::high_resolution_clock::now();
            opt.optimize_reduce(rastrigin, grid, execution::par,Tp,-1);
            end = std::chrono::high_resolution_clock::now();
            tempi_par_reduce[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
        } 
    }

    }
    //print
    for(int gs = 0; gs< grid_sizes.size(); gs++){
        if(n_thread == 2){
            std::cout<<"opt sequenziale "<<grid_sizes[gs]<<" ";
            for(auto& t : tempi_seq[gs]){std::cout<<t.count()<<" ";}
            std::cout<<std::endl;
        }
        std::cout<<"opt_parallel thread_"<<n_thread<<" "<<grid_sizes[gs]<<" ";
        for(auto& t : tempi_par[gs]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
        std::cout<<"opt_par_variadic thread_"<<n_thread<<" "<<grid_sizes[gs]<<" ";
        for(auto& t : tempi_par_graninput[gs]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
        std::cout<<"opt_par_reduce thread_"<<n_thread<<" "<<grid_sizes[gs]<<" ";
        for(auto& t : tempi_par_reduce[gs]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    
}

// // sequenziale vs parallelo con gran default (-1) vs granuarity t.c 10 job per worker
// {//==================== GRID SEARCH ==============================================
//     // default granularity
//     auto start = std::chrono::high_resolution_clock::now();
//     auto end = std::chrono::high_resolution_clock::now();
//     std::vector<int> grid_sizes = {12800000}; //
//     std::vector<int> runs_vett = {10};//
//     std::vector<std::vector<std::chrono::microseconds>> tempi_seq(grid_sizes.size()); // vect esterno un elemento per ogni grid_size, quelli interni sono tempi di runs
//     std::vector<std::vector<std::chrono::microseconds>> tempi_par(grid_sizes.size());
//     std::vector<std::vector<std::chrono::microseconds>> tempi_par_graninput(grid_sizes.size());
//     std::vector<bool> samesame;
//     double lower = -5;
//     double upper = 5;
//     std::uniform_real_distribution<double> unif(lower,upper);
//     std::default_random_engine re;    
//     //rastrigin function
//     fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 20 + p[0]*p[0] + p[1]*p[1] - 10*std::cos(2*M_PI*p[0]) - 10*std::cos(2*M_PI*p[1]); })> rastrigin;
//     //loop per diverse size di griglia in grid_sizes
//     for(int gs = 0; gs< grid_sizes.size(); gs++){
//         //crea griglia
//         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> grid;
//         grid.resize(grid_sizes[gs],2);
//         for(int i =0; i<grid.rows();++i){
//             grid(i,0) = unif(re);
//             grid(i,1) = unif(re);
//         } 
//     {//seq
//         if(n_thread == 2){
//             for (int run = 0; run<runs_vett[gs]; run ++){
//                 fdapde::GridSearch<2> opt;
//                 start = std::chrono::high_resolution_clock::now();
//                 opt.optimize(rastrigin, grid);
//                 end = std::chrono::high_resolution_clock::now();
//                 tempi_seq[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
//             }
//         }
//     }
//     {//parallel
//         for (int run = 0; run<runs_vett[gs]; run ++){
//             fdapde::threadpool<fdapde::steal::random> Tp(1024, n_thread);
//             fdapde::GridSearch<2> opt;
//             start = std::chrono::high_resolution_clock::now();
//             opt.optimize(rastrigin, grid, execution::par,Tp,-1);
//             end = std::chrono::high_resolution_clock::now();
//             tempi_par[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
//         }
//     }
//     {//parallel
//         for (int run = 0; run<runs_vett[gs]; run ++){
//             fdapde::threadpool<fdapde::steal::random> Tp(1024, n_thread);
//             fdapde::GridSearch<2> opt;
//             start = std::chrono::high_resolution_clock::now();
//             opt.optimize(rastrigin, grid, execution::par,Tp,grid_sizes[gs]/(n_thread*10)); // paralle con granularity t.c. 10 job per worker (circa)
//             end = std::chrono::high_resolution_clock::now();
//             tempi_par_graninput[gs].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
//         } 
//     }
//     }
//     //print
//     for(int gs = 0; gs< grid_sizes.size(); gs++){
//         if(n_thread == 2){
//             std::cout<<"opt sequenziale "<<grid_sizes[gs]<<" ";
//             for(auto& t : tempi_seq[gs]){std::cout<<t.count()<<" ";}
//             std::cout<<std::endl;
//         }
//         std::cout<<"opt_par_grandefault thread_"<<n_thread<<" "<<grid_sizes[gs]<<" ";
//         for(auto& t : tempi_par[gs]){std::cout<<t.count()<<" ";}
//         std::cout<<std::endl;
//         std::cout<<"opt_par_grantc10jpw thread_"<<n_thread<<" "<<grid_sizes[gs]<<" ";
//         for(auto& t : tempi_par_graninput[gs]){std::cout<<t.count()<<" ";}
//         std::cout<<std::endl;
//     }
    
// }

return 0;
}