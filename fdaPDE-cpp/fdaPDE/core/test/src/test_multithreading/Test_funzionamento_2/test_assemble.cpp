#include <fdaPDE/finite_elements.h>
#include<fdaPDE/multithreading.h>
using namespace fdapde;

int main(int argc, char** argv){
    int nodi = std::stoi(argv[1]);
    int workers = std::stoi(argv[2]);
    int kk = std::stoi(argv[3]);
    fdapde::threadpool Tp(1000,workers);
    Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodi);//cube

    FeSpace Vh(unit_square, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
    auto b = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
    
//cronometro assemblaggio non parallello
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::SparseMatrix<double> A = a.assemble();//(execution::par); // use parallel version

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<"tempo (microsec) assemblaggio non parallelo: "<<duration.count()<<std::endl; 
//cronometro assemblaggio parallelo
    auto start2 = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> A2 = b.assemble(execution::par,Tp,kk); // use parallel version
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    std::cout<<"tempo (microsec) assemblaggio parallelo, thread: "<<workers<<" ,"<<duration2.count()<<std::endl;  
    

    std::cout<<A.size()<<std::endl;
    std::cout<<A2.size()<<std::endl;
    //std::cout<<A<<std::endl;
    //std::cout<<A2<<std::endl;
    if (A.isApprox(A2, 0.000000000000001)) {
        std::cout << "Le matrici sono identiche." << std::endl; 
    }else{
        std::cout << "Le matrici sono diverse." << std::endl;
    }

    return 0;
}

/*
    std::cout<<A<<std::endl;
    std::cout<<A.size()<<std::endl;
    int k;
    for (k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            std::cout << "A(" << it.row() << "," << it.col() << ") = " << it.value() << "\n";
        }
    }
    std::cout<<k<<std::endl;

*/
