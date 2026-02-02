#include<fdaPDE/multithreading.h>
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
void random_op_di_n_elem(Q& q,int n, std::vector<int>& opr, int j){
    std::string el = "ciao";
    for (int i = 0; i<n;++i) {
        switch (opr[i+j*n]) {
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

int main(int argc, char** argv){
    int n_thread= std::stoi(argv[1]);
    int runs = 20;
/*  
{//================= Test threadpool send-steal ==============================
    runs = 30;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::chrono::microseconds>> tempi_send_steal(6); // 2x3 mostfree-mostbusy, mostfree-rando, mostfree-ranhalf,...
    int n_job = 4096; //evitiamo di riempire
    int n_op_per_job = 5000;
    std::vector<std::future<void>> futs;
    for(int run= 0; run<runs; run ++){
        {// mostFree-mostBusy 0 
            fdapde::Threadpool<fdapde::steal::most_busy> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        } 
        {// mostFree-random 1 
            fdapde::Threadpool<fdapde::steal::random> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        }
        {// mostFree-randomHalf 2 
            fdapde::Threadpool<fdapde::steal::random_half_most_busy> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        }
        {// round-mostbusy 3 
            fdapde::Threadpool<fdapde::steal::most_busy> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        }
        {// round-random 4 
            fdapde::Threadpool<fdapde::steal::random> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[4].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        }
        {// round-randomHalf 5 
            fdapde::Threadpool<fdapde::steal::random_half_most_busy> tp(4096,n_thread);
            start = std::chrono::high_resolution_clock::now();
            for( int i =0; i<n_job; i++){
                futs.push_back(std::move(tp.send_task_mostfree([=](int i)mutable{
                    if(i%2 == 0){ //job pari sono il doppio
                        n_op_per_job *= 4;
                    }
                    int a= 0;
                    for (int j = 0; j<n_op_per_job; j++){a++;}},i)));}
            for(int i= 0; i<n_job; i++){futs[i].get();}
            auto end = std::chrono::high_resolution_clock::now();
            tempi_send_steal[5].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));     
            futs.clear();
        }
    }
    std::cout<<std::endl;
    std::cout<<"===========TEST THREADPOOL SEND_STEAL======================================================================================================================================================================================"<<std::endl;
    std::cout<<"======= send-steal test con job sbilanciati (1/2 hanno n_op_per_job, 1/2 hanno n_op_per_job*4) ================================================================================================================================================================================="<<std::endl;
    std::cout<<"===== n_job: "<<n_job<<", n_op_per_job: "<<n_op_per_job<<", Threadpool: sizequeue(1024),n_thread("<<n_thread<<") ==========================================================================================================="<<std::endl;
    std::vector<std::string> tag_coppie_send_steal = {"mostfree-mostbusy","mostfree-random","mostfree-randomHalf","round-mostbusy","round-random","round-randomHalf"};
    for (int i = 0; i<6; i++){
        std::cout<<tag_coppie_send_steal[i]<<": ";
        for(auto& t: tempi_send_steal[i]){std::cout<<t.count()<<" ,";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
}
*/
{//============= TEST PARALLE FOR SUM MATRIX===============================================================
    runs = 20;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::chrono::microseconds>> tempi_for(3); // 0 for, 1 parallel_for, 2 openMP
    std::vector<std::string> tag_tipi_for = {"For","Parallel_for","OpenMp"};
    int size = 8000;
    std::vector<std::vector<int>> A;
    std::vector<std::vector<int>> B;
    std::vector<std::vector<int>> C;
    for (int i = 0; i<size; i++){
        A.emplace_back(size,1);
        B.emplace_back(size,1);
        C.emplace_back(size,0);
    }
    fdapde::Threadpool<fdapde::steal::most_busy> tp(1024,n_thread);
    if(n_thread == 2){
        //for
        for(int run = 0; run <runs; run ++){
            start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i<size; i++){
                for(int j=0; j<size; j++){C[i][j] = A[i][j] + B[i][j];}}
            end = std::chrono::high_resolution_clock::now();
            tempi_for[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start)); 
        }
    }
    //parallel_for, gran default -1
    for(int run = 0; run <runs; run ++){
        start = std::chrono::high_resolution_clock::now();
        tp.parallel_for(0,size,[&](int i){
            for(int j=0; j<size; j++){
                C[i][j] = A[i][j] + B[i][j];
            }
        },25);
        end = std::chrono::high_resolution_clock::now();
        tempi_for[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start)); 
    }
    // 2 openMp
    for(int run = 0; run <runs; run ++){
        start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(n_thread) shared(A,B,C)
        for(int i = 0; i<size; i++){
            for(int j=0; j<size; j++){
                C[i][j] = A[i][j] + B[i][j];}}
        end = std::chrono::high_resolution_clock::now();
        tempi_for[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start)); 
    }

    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"========================================================================================================================================================================================"<<std::endl;
    std::cout<<"===========TEST PARALLEL_FOR SUM MATRIX ELEMENT-WISE, granularity 25======================================================================================================================================================================================"<<std::endl;
    std::cout<<"========================================================================================================================================================================================"<<std::endl;
    std::cout<<"===== size matrix: NxN con N = "<<size<<", Threadpool: sizequeue(1024),n_thread("<<n_thread<<"), "<<"Granularity di parallel_for di default: -1"<<" ==========================================================================================================="<<std::endl;
    int inizio = (n_thread == 2)? 0:1; //stampa for solo se n_thread ==2 
    for (int i = inizio; i<3; i++){
        std::cout<<tag_tipi_for[i]<<": ";
        for(auto& t: tempi_for[i]){std::cout<<t.count()<<" ,";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
}
/*
{//============== Test deque vs synchro_queue =====================
    runs = (n_thread<16)? 100:50;
    int tot_elem = 64000;
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
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<runs; i++){
    if(n_thread == 2){//calcola tempi singoli solo in n_thread == 2
        {//single-deque
            using queue = Worker_queue_deque<std::string>;
            queue d;
            //push_front 0
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,0> (std::ref(d),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            d_single[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_front 2
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,2> (std::ref(d),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            d_single[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //push_back 1
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,1> (std::ref(d),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            d_single[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_back 3
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,3> (std::ref(d),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            d_single[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
        }
        {//single-synchro Relax
            using queue = fdapde::synchro_queue<std::string,fdapde::relax>;
            queue q(tot_elem);
            //push_front 0
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,0> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_r_single[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_front 2
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,2> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_r_single[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //push_back 1
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,1> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_r_single[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_back 3
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,3> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_r_single[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
        }
        {//single-synchro Hold no wait
            using queue = fdapde::synchro_queue<std::string,fdapde::hold_nowait>;
            queue q(tot_elem);
            //push_front 0
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,0> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_h_single[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_front 2
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,2> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_h_single[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //push_back 1
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,1> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_h_single[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_back 3
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,3> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_h_single[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
        }
        {//single-synchro Hold wait
            using queue = fdapde::synchro_queue<std::string,fdapde::hold_wait>;
            queue q(tot_elem);
            //push_front 0
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,0> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_hw_single[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_front 2
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,2> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_hw_single[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //push_back 1
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,1> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_hw_single[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
            //pop_back 3
            start = std::chrono::high_resolution_clock::now();
            op_di_n_elem<queue,3> (std::ref(q),tot_elem);
            end = std::chrono::high_resolution_clock::now();
            s_hw_single[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));  
        }
    }
        {// multi-deque
            using queue = Worker_queue_deque<std::string>;
            queue q;
            std::vector<std::thread> thread_pool;
            //push front 0
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,0>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            d_multi[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop front 2
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,2>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            d_multi[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //push back 1
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,1>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            d_multi[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop back 3
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,3>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            d_multi[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
        }
        {// multi-hold_npwait
            using queue = fdapde::synchro_queue<std::string,fdapde::hold_nowait>;
            queue q(tot_elem);
            std::vector<std::thread> thread_pool;
            //push front 0
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,0>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_h_multi[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop front 2
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,2>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_h_multi[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //push back 1
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,1>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_h_multi[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop back 3
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,3>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_h_multi[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
        }
        {// multi-relax
            using queue = fdapde::synchro_queue<std::string,fdapde::relax>;
            queue q(tot_elem);
            std::vector<std::thread> thread_pool;
            //push front 0
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,0>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_r_multi[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop front 2
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,2>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_r_multi[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //push back 1
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,1>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_r_multi[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop back 3
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,3>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_r_multi[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
        }
        {// multi-hold_wait
            using queue = fdapde::synchro_queue<std::string,fdapde::hold_wait>;
            queue q(tot_elem);
            std::vector<std::thread> thread_pool;
            //push front 0
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,0>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_hw_multi[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop front 2
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,2>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_hw_multi[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //push back 1
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,1>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_hw_multi[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //pop back 3
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(op_di_n_elem<queue,3>,std::ref(q),elem_per_thread);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            s_hw_multi[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
        }
        {// multi-RANDOM
            using d = Worker_queue_deque<std::string>;
            using q_r = fdapde::synchro_queue<std::string,fdapde::relax>;
            using q_h = fdapde::synchro_queue<std::string,fdapde::hold_nowait>;
            using q_hw = fdapde::synchro_queue<std::string,fdapde::hold_wait>;
            d d_;
            q_r q_r_(2*tot_elem);
            q_h q_h_(2*tot_elem);
            q_hw q_hw_(2*tot_elem);
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
            std::vector<int> randomop;
            std::mt19937 gen(42);
            std::uniform_int_distribution<> distrib(0,3);
            for(int l = 0; l<tot_elem; l++){
                randomop.push_back(distrib(gen));
            }
            //deque
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(random_op_di_n_elem<d>,std::ref(d_),elem_per_thread,std::ref(randomop),j);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            tempi_random[0].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            //relax
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(random_op_di_n_elem<q_r>,std::ref(q_r_),elem_per_thread,std::ref(randomop),j);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            tempi_random[1].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            // hold
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(random_op_di_n_elem<q_h>,std::ref(q_h_),elem_per_thread,std::ref(randomop),j);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            tempi_random[2].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();
            // hold_wait
            start = std::chrono::high_resolution_clock::now(); 
            for (int j=0; j<n_thread; j++){thread_pool.emplace_back(random_op_di_n_elem<q_hw>,std::ref(q_hw_),elem_per_thread,std::ref(randomop),j);}
            for (int j= 0; j<n_thread; j++){thread_pool[j].join();}
            end = std::chrono::high_resolution_clock::now();
            tempi_random[3].push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start));
            thread_pool.clear();        
        }
    }
    std::cout<<"===========TEST DEQUE VS sYNCHRO_QUEUE======================================================================================================================================================================================"<<std::endl;
    std::cout<<"=============== tot_elem: "<<tot_elem<<"===================================================================================================="<<std::endl;
    std::vector<std::string> names = {"Push_Front:","Push_Back:","Pop_Front:","Pop_Back:"};
if(n_thread == 2){
    std::cout<<"===============tempi single================================================================================================="<<std::endl;
    std::cout<<"deque"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : d_single[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_relax"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_r_single[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_hold"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_h_single[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_hold_wait"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_hw_single[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
}
    std::cout<<"=========tempi multi, n_thread: "<<n_thread<<" ================================================================================================="<<std::endl;
    std::cout<<"deque"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : d_multi[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_relax"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_r_multi[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_hold"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_h_multi[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"S_hold_wait"<<std::endl;
    for(size_t name = 0; name < names.size(); name++ ){
        std::cout<<names[name];
        for (auto& t : s_hw_multi[name]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<"=========tempi Random multi, n_thread: "<<n_thread<<" ================================================================================================="<<std::endl;
    std::vector<std::string> nomi_code ={"deque:","relax:","hold:","hold_wait:"};
    for(int i = 0; i<4; i++){
        std::cout<<nomi_code[i];
        for (auto& t : tempi_random[i]){std::cout<<t.count()<<", ";}
        std::cout<<std::endl;
        std::cout<<std::endl;

    }

}
*/
return 0;
}
