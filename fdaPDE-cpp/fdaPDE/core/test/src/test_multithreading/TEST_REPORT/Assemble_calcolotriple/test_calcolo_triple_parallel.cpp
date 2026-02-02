#include <fdaPDE/finite_elements.h>
#include<fdaPDE/multithreading.h>
//metti in assemble bilinear form il cronometro e output da li
using namespace fdapde;

int main(int argc, char** argv){
    int nodi = std::stoi(argv[1]);
    int workers = std::stoi(argv[2]);
    int granularity = std::stoi(argv[3]);
    fdapde::Threadpool<fdapde::steal::random> Tp(1024,workers);
    Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodi);

    FeSpace Vh(unit_square, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);
    auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form

    Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,granularity); // use parallel version
    

    return 0;
}