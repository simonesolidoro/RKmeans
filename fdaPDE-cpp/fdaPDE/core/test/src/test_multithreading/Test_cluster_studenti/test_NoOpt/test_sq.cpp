#include<fdaPDE/multithreading.h>
auto start = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

template<typename Q, int op>// 0 push front, 1 push back, 2 pop front, 3 pop back
void op_di_n_elem(Q & q,int n){
    if constexpr(op == 0){
        std::string val = "ciao";
        for (int j=0; j<n; j++){
            q.push_front(val);
        }
    }
    if constexpr(op == 1){
        std::string val = "ciao";
        for (int j=0; j<n; j++){
            q.push_back(val);
        }
    }
    if constexpr(op == 2){
        for (int j=0; j<n; j++){
            q.pop_front();
        }
    }
    if constexpr(op == 3){
        for (int j=0; j<n; j++){
            q.pop_back();
        }
    }
};
template<typename Q>
void random_op_di_n_elem(Q& q, const std::vector<int>& opr){
    std::string el = "ciao";
    for (int i = 0; i<opr.size();++i) {
        switch (opr[i]) {
        case 0:
            q.push_front(el);
            break;
        case 1:
            q.push_back(el);
            break;
        case 2:
            q.pop_front();
            break;
        case 3:
            q.pop_back();
            break;
        }
    }
}
template<typename queue, int a>
void test_single(std::vector<std::vector<std::chrono::microseconds>>& time, queue& d, int tot_elem){
    start = std::chrono::high_resolution_clock::now();
    op_di_n_elem<queue,a> (std::ref(d),tot_elem);
    end = std::chrono::high_resolution_clock::now();
    time[a].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
} 
template<typename queue, int a>
void test_multi(std::vector<std::thread>& thread_pool, int n_thread,int elem_per_thread, std::vector<std::vector<std::chrono::microseconds>>& time, queue& q){
    start = std::chrono::high_resolution_clock::now(); 
    for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,a>,std::ref(q),elem_per_thread);}
    for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
    end = std::chrono::high_resolution_clock::now();
    time[a].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    thread_pool.clear();  
} 
template<typename queue>
void test_random(std::vector<std::thread>& thread_pool, int n_thread, std::vector<std::vector<std::chrono::microseconds>>& time, queue& d, std::vector<std::vector<int>>& randomop){
    int a;
    if constexpr(std::is_same_v<queue,fdapde::synchro_queue<std::string,fdapde::relaxed>>){a = 1;}
    if constexpr(std::is_same_v<queue,fdapde::synchro_queue<std::string,fdapde::deferred>>){a = 2;}
    if constexpr(std::is_same_v<queue,fdapde::synchro_queue<std::string,fdapde::blocking>>){a = 3;}
    if constexpr(std::is_same_v<queue,Worker_queue_deque<std::string>>){a = 0;}
    start = std::chrono::high_resolution_clock::now(); 
    for (int j=0; j<n_thread; j++){thread_pool.emplace_back(random_op_di_n_elem<queue>,std::ref(d),std::ref(randomop[j]));}
    for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
    end = std::chrono::high_resolution_clock::now();
    time[a].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
    thread_pool.clear(); 
} 

template<typename queue>
void riempi(queue& d, int tot_elem){
    for (int i = 0; i<tot_elem; i++){
        d.push_back("ciao");
    } 
} 
int main(int argc, char** argv){
    int n_thread= std::stoi(argv[1]);
    int tot_elem = std::stoi(argv[2]);
    int runs = std::stoi(argv[3]);
  
{//============== Test deque vs synchro_queue =====================
    int elem_per_thread = tot_elem / n_thread;
    //tempi single
    std::vector<std::vector<std::chrono::microseconds>> d_single(4); //per ogni operazione {1,2,3,4} un vettore di tempi
    std::vector<std::vector<std::chrono::microseconds>> s_r_single(4);
    std::vector<std::vector<std::chrono::microseconds>> s_h_single(4);
    std::vector<std::vector<std::chrono::microseconds>> s_hw_single(4);
    //tempi multi
    std::vector<std::vector<std::chrono::microseconds>> d_multi(4); //per ogni operazione {1,2,3,4} un vettore di tempi
    std::vector<std::vector<std::chrono::microseconds>> s_r_multi(4);
    std::vector<std::vector<std::chrono::microseconds>> s_h_multi(4);
    std::vector<std::vector<std::chrono::microseconds>> s_hw_multi(4);
    // tempi random
    std::vector<std::vector<std::chrono::microseconds>> tempi_random(4);
    for(int i = 0; i<runs; i++){
        if(n_thread == 2){//calcola tempi singoli solo in n_thread == 2
            //single
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            //push_front 0
            test_single<Worker_queue_deque<std::string>,0>(d_single, q_deque, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::relaxed>,0>(s_r_single, q_relaxed, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::deferred>,0>(s_h_single, q_deferred, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::blocking>,0>(s_hw_single, q_blocking, tot_elem);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            riempi(q_deque,tot_elem);
            riempi(q_relaxed,tot_elem);
            riempi(q_deferred,tot_elem);
            riempi(q_blocking,tot_elem);
            //pop_front 2
            test_single<Worker_queue_deque<std::string>,2>(d_single, q_deque, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::relaxed>,2>(s_r_single, q_relaxed, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::deferred>,2>(s_h_single, q_deferred, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::blocking>,2>(s_hw_single, q_blocking, tot_elem);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            //push_back 1
            test_single<Worker_queue_deque<std::string>,1>(d_single, q_deque, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::relaxed>,1>(s_r_single, q_relaxed, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::deferred>,1>(s_h_single, q_deferred, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::blocking>,1>(s_hw_single, q_blocking, tot_elem);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            riempi(q_deque,tot_elem);
            riempi(q_relaxed,tot_elem);
            riempi(q_deferred,tot_elem);
            riempi(q_blocking,tot_elem);
            //pop_back 3
            test_single<Worker_queue_deque<std::string>,3>(d_single, q_deque, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::relaxed>,3>(s_r_single, q_relaxed, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::deferred>,3>(s_h_single, q_deferred, tot_elem);
            test_single<fdapde::synchro_queue<std::string,fdapde::blocking>,3>(s_hw_single, q_blocking, tot_elem);
            }
        }
        {// multi
            std::vector<std::thread> thread_pool;
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);

            //push_front 0
            test_multi<Worker_queue_deque<std::string>,0>(thread_pool,n_thread,elem_per_thread,d_multi, q_deque);
            test_multi<fdapde::synchro_queue<std::string,fdapde::relaxed>,0>(thread_pool,n_thread,elem_per_thread,s_r_multi, q_relaxed);
            test_multi<fdapde::synchro_queue<std::string,fdapde::deferred>,0>(thread_pool,n_thread,elem_per_thread,s_h_multi, q_deferred);
            test_multi<fdapde::synchro_queue<std::string,fdapde::blocking>,0>(thread_pool,n_thread,elem_per_thread,s_hw_multi, q_blocking);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            riempi(q_deque,tot_elem);
            riempi(q_relaxed,tot_elem);
            riempi(q_deferred,tot_elem);
            riempi(q_blocking,tot_elem);
            //pop_front 2
            test_multi<Worker_queue_deque<std::string>,2>(thread_pool,n_thread,elem_per_thread,d_multi, q_deque);
            test_multi<fdapde::synchro_queue<std::string,fdapde::relaxed>,2>(thread_pool,n_thread,elem_per_thread,s_r_multi, q_relaxed);
            test_multi<fdapde::synchro_queue<std::string,fdapde::deferred>,2>(thread_pool,n_thread,elem_per_thread,s_h_multi, q_deferred);
            test_multi<fdapde::synchro_queue<std::string,fdapde::blocking>,2>(thread_pool,n_thread,elem_per_thread,s_hw_multi, q_blocking);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);

            //push_back 1
            test_multi<Worker_queue_deque<std::string>,1>(thread_pool,n_thread,elem_per_thread,d_multi, q_deque);
            test_multi<fdapde::synchro_queue<std::string,fdapde::relaxed>,1>(thread_pool,n_thread,elem_per_thread,s_r_multi, q_relaxed);
            test_multi<fdapde::synchro_queue<std::string,fdapde::deferred>,1>(thread_pool,n_thread,elem_per_thread,s_h_multi, q_deferred);
            test_multi<fdapde::synchro_queue<std::string,fdapde::blocking>,1>(thread_pool,n_thread,elem_per_thread,s_hw_multi, q_blocking);
            }
            {
            Worker_queue_deque<std::string> q_deque;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_relaxed(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_deferred(tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_blocking(tot_elem);
            riempi(q_deque,tot_elem);
            riempi(q_relaxed,tot_elem);
            riempi(q_deferred,tot_elem);
            riempi(q_blocking,tot_elem);
            //pop_back 3
            test_multi<Worker_queue_deque<std::string>,3>(thread_pool,n_thread,elem_per_thread,d_multi, q_deque);
            test_multi<fdapde::synchro_queue<std::string,fdapde::relaxed>,3>(thread_pool,n_thread,elem_per_thread,s_r_multi, q_relaxed);
            test_multi<fdapde::synchro_queue<std::string,fdapde::deferred>,3>(thread_pool,n_thread,elem_per_thread,s_h_multi, q_deferred);
            test_multi<fdapde::synchro_queue<std::string,fdapde::blocking>,3>(thread_pool,n_thread,elem_per_thread,s_hw_multi, q_blocking);
            }
        }
        {// multi-RANDOM
            Worker_queue_deque<std::string> d_;;
            fdapde::synchro_queue<std::string,fdapde::relaxed> q_r_(2*tot_elem);
            fdapde::synchro_queue<std::string,fdapde::deferred> q_h_(2*tot_elem);
            fdapde::synchro_queue<std::string,fdapde::blocking> q_hw_(2*tot_elem);
            
            //riempiamo a metà direi che basta e avanza tanto operazioni da distrib unif impossibile riempire o svuotare
            std::string elem = "ciao";
            for (int i = 0; i<tot_elem; i++){
                d_.push_front(elem);
                q_r_.push_front(elem);
                q_h_.push_front(elem);
                q_hw_.push_front(elem);
            }
            std::vector<std::thread> thread_pool;
            //generatore numeri casuali cosi per tutti uguale
            std::vector<std::vector<int>> randomop(n_thread);
            std::mt19937 gen(42);
            std::uniform_int_distribution<> distrib(0,3);
            for(int j = 0; j< n_thread; j++){
                for(int l = 0; l<elem_per_thread; l++){
                    randomop[j].push_back(distrib(gen));
                }
            }
            test_random(thread_pool, n_thread, tempi_random,d_, randomop);
            test_random(thread_pool, n_thread, tempi_random,q_r_, randomop);
            test_random(thread_pool, n_thread, tempi_random,q_h_, randomop);
            test_random(thread_pool, n_thread, tempi_random,q_hw_, randomop);
        }
    }
    //std::cout<<"===========TEST DEQUE VS sYNCHRO_QUEUE======================================================================================================================================================================================"<<std::endl;
    //std::cout<<"=============== tot_elem: "<<tot_elem<<"===================================================================================================="<<std::endl;
    std::vector<std::string> names = {"Push_Front","Push_Back","Pop_Front","Pop_Back"};
if(n_thread == 2){
    //std::cout<<"===============tempi single================================================================================================="<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"deque "<<names[name]<<" single ";
        for (auto& t : d_single[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"relaxed "<<names[name]<<" single ";
        for (auto& t : s_r_single[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"deferred "<<names[name]<<" single ";
        for (auto& t : s_h_single[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"blocking "<<names[name]<<" single ";
        for (auto& t : s_hw_single[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    
}
    //std::cout<<"=========tempi multi, n_thread: "<<n_thread<<" ================================================================================================="<<std::endl;
        for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"deque "<<names[name]<<" multi_"<<n_thread<<" ";
        for (auto& t : d_multi[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"relaxed "<<names[name]<<" multi_"<<n_thread<<" ";
        for (auto& t : s_r_multi[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"deferred "<<names[name]<<" multi_"<<n_thread<<" ";
        for (auto& t : s_h_multi[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<"blocking "<<names[name]<<" multi_"<<n_thread<<" ";
        for (auto& t : s_hw_multi[name]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }
    //std::cout<<"=========tempi Random multi, n_thread: "<<n_thread<<" ================================================================================================="<<std::endl;
    std::vector<std::string> nomi_code ={"deque","relaxed","deferred","blocking"};
    for(int i = 0; i<4; i++){
        std::cout<<nomi_code[i]<<" random multi_"<<n_thread<<" ";
        for (auto& t : tempi_random[i]){std::cout<<t.count()<<" ";}
        std::cout<<std::endl;
    }

}
return 0;
}

