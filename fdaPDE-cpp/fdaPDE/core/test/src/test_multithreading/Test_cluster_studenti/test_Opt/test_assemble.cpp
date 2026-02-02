#include<fdaPDE/multithreading.h>
#include<fdaPDE/optimization.h>
#include <fdaPDE/finite_elements.h>


int main(int argc, char** argv){
    int n_thread= std::stoi(argv[1]);

/*
// effetto di granularity in solo calcolo triple seq vs gran def parallel vs gran t.c 10jpw
{// ============================= ASSEMBLE =============================================
    using namespace fdapde;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> nodes = {1000}; // cosi da avere anche scalabilità debole con 8 16 32 oppure 16 32 64
    std::vector<int> runs_vett = {3};//{20,20,20};//{20,20,20};//{1,1,1};
    std::vector<std::vector<std::chrono::microseconds>> tempo_seq_assemble(nodes.size()); //per ogni nodi vettore di tempi di assemblaggio completo. (vediamo se overhead in setfromtriple rimane in cluster (speriamo di no perchè il vttore di triple è uguale in seq e in par))
    std::vector<std::vector<std::chrono::microseconds>> tempo_par_assemble(nodes.size());
    if(n_thread == 2){
        //std::cout<<"=====Tempi SEQUENZIALE========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple "<<nodes[nodi]<<" sequenziale ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple();// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }            
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grandef "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,-1);// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }
    }
        {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grantc10jpw "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,((nodes[nodi]-1)*(nodes[nodi]-1)*2)/(n_thread*10));// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }

    }
*/

    // seq vs gran1 vs gran_input vs gran_input_iterator con default granularity -1
{// ============================= ASSEMBLE =============================================
    using namespace fdapde;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> nodes = {500}; // cosi da avere anche scalabilità debole con 8 16 32 oppure 16 32 64
    std::vector<int> runs_vett = {3};//{20,20,20};//{20,20,20};//{1,1,1};
    std::vector<std::vector<std::chrono::microseconds>> tempo_seq_assemble(nodes.size()); //per ogni nodi vettore di tempi di assemblaggio completo. (vediamo se overhead in setfromtriple rimane in cluster (speriamo di no perchè il vttore di triple è uguale in seq e in par))
    std::vector<std::vector<std::chrono::microseconds>> tempo_par_assemble(nodes.size());
    std::vector<std::vector<std::chrono::microseconds>> tempo_par_iter_assemble(nodes.size());
    if(n_thread == 2){
        //std::cout<<"=====Tempi SEQUENZIALE========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple "<<nodes[nodi]<<" sequenziale ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple();// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }            
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grandef "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,-1);// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_graninput_iterator "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_iterator_tempotriple(execution::par,Tp,-1);// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_graninput "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_graninput_tempotriple(execution::par,Tp,-1);// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }

    }
/*
// effetto di granularity in solo calcolo triple seq vs parallel con granularity t.c. 1 jpw , 16 job total, 160 job total, 640 job total 
{// ============================= ASSEMBLE =============================================
    using namespace fdapde;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> nodes = {1000}; // cosi da avere anche scalabilità debole con 8 16 32 oppure 16 32 64
    std::vector<int> runs_vett = {3};//{20,20,20};//{20,20,20};//{1,1,1};
    std::vector<std::vector<std::chrono::microseconds>> tempo_seq_assemble(nodes.size()); //per ogni nodi vettore di tempi di assemblaggio completo. (vediamo se overhead in setfromtriple rimane in cluster (speriamo di no perchè il vttore di triple è uguale in seq e in par))
    std::vector<std::vector<std::chrono::microseconds>> tempo_par_assemble(nodes.size());
    if(n_thread == 2){
        //std::cout<<"=====Tempi SEQUENZIALE========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple "<<nodes[nodi]<<" sequenziale ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple();// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }            
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grandef "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,-1);// il cout dei tempi è dentro assemble() con start e end che racchiudono assemble(...)
            }
            std::cout<<std::endl;
        }
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grantc16job "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,((nodes[nodi]-1)*(nodes[nodi]-1)*2)/(16));// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grantc160job "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,((nodes[nodi]-1)*(nodes[nodi]-1)*2)/(160));// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }
    {
        //std::cout<<"=====Tempi PARALLELO n_thread: "<<n_thread<<"========================================================================================================================================================================================================================================================="<<std::endl; 
        for(int nodi = 0; nodi<nodes.size(); nodi++){
            Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodes[nodi]);
            FeSpace Vh(unit_square, P1<1>);
            TrialFunction u(Vh);
            TestFunction  v(Vh);
            auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
            std::cout<<"calcolo_triple_grantc640jpw "<<nodes[nodi]<<" thread_"<<n_thread<<" ";
            for(int run = 0; run <runs_vett[nodi]; run ++){
                fdapde::threadpool<fdapde::steal::random> Tp(1024,n_thread);
                Eigen::SparseMatrix<double> A = a.assemble_tempotriple(execution::par,Tp,((nodes[nodi]-1)*(nodes[nodi]-1)*2)/(640));// numero celle (e quindi numero iterazioni range)= (nodiperlato-1)^2*2
            }
            std::cout<<std::endl;
        }
    }
    }
*/
return 0;
}

