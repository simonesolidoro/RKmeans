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

#include<fdaPDE/multithreading.h>

int main(int argc,char** argv)
{//parallel_for_sure
    int gran = std::stoi(argv[1]);
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(64,8);
    std::atomic<int> a=0; //usata per verifica tutti jo vengano eseguiti (a deve arrivare ad n)

    std::vector<int> V(1000,1); //{1,2,3,4,5,6,7,8,9,0};
    std::list<int> W(1000,1); // {1,2,3,4,5,6,7,8,9};

    tp.parallel_for(V.begin(),V.end(),[&](std::vector<int>::iterator iter){
        //std::cout<<*iter<<" da thread: "<<std::this_thread::get_id()<<std::endl;
        a++;
    });

    std::cout<<"a arriavta da 0 ad: "<<a.load()<<" doveva arrivare a:"<<V.size()<<std::endl;
    a.store(0);
    std::cout<<"-------------------------------------- granularity random acces ----------------------------------------"<<std::endl;
    int tmp = 10;
    tp.parallel_for(V.begin(),V.end(),[&](std::vector<int>::iterator iter, int index_w, int& tmp){
        //std::cout<<*iter<<" da thread: "<<std::this_thread::get_id()<<std::endl;
        a++;
    },gran,tmp);
    std::cout<<"a arriavta da 0 ad: "<<a.load()<<" doveva arrivare a:"<<V.size()<<std::endl;
    
    std::cout<<"-------------------------------------- granularity non random acces (list,map,set)----------------------------------------"<<std::endl;
    a.store(0);
    tp.parallel_for(W.begin(),W.end(),[&](std::list<int>::iterator iter, int index_w, int& tmp){
       //std::cout<<*iter<<" da thread: "<<std::this_thread::get_id()<<std::endl;
        a++;
    },gran,tmp);
    std::cout<<"a arriavta da 0 ad: "<<a.load()<<" doveva arrivare a:"<<V.size()<<std::endl;
    /*

//for non parallel
    a.store(0);
    auto start2 = std::chrono::high_resolution_clock::now();

    for (int i = 0;i < n;i++){
        a++;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    std::cout<<"for - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration2.count()<< " microsecondi\n";
    

// parallel_for_sure_granularity
    a.store(0);
    int n_it = std::stoi(argv[1]); //numero di blocchi in cui dividere range di for
    auto start3 = std::chrono::high_resolution_clock::now();
 
    tp.parallel_for_sure_granularity(0,10000,n_it,[&](int i){a++;
        //std::cout<<i<<std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    });
    
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    std::cout<<"par_for_sure_n - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi con n_it: "<<n_it<<std::endl;

*/
return 0;
}
