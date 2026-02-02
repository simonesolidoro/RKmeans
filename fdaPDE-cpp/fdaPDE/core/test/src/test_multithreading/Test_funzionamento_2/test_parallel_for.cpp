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

int main(int argc,char** argv){
    int start = std::stoi(argv[1]);
    int end = std::stoi(argv[2]);
    int granularity = std::stoi(argv[3]);
    int n = end-start;
    fdapde::threadpool<fdapde::steal::random> tp(1024,8);

{
std::cout<<"================ parallel_for_granularity_variadic  ================"<<std::endl; 
    std::atomic<int> a=0; //usata per verifica tutti job vengano eseguiti (a deve arrivare ad n)
    int tmp_int = 10;
    tp.parallel_for(start,end,[&](int i,int index_w, int& tmp){
        a++;
    },granularity,tmp_int);
    if(a.load() == n){std::cout<<"variadic ok "<<std::endl;}
    else{std::cout<<"qualcosa non va :("<<std::endl;}
}
{
std::cout<<"================ parallel_for (gran=1) ================"<<std::endl;
    std::atomic<int> a=0; //usata per verifica tutti job vengano eseguiti (a deve arrivare ad n)
    tp.parallel_for(start,end,[&](int i){
        a++;
    });
    if(a.load() == n){std::cout<<" ok "<<std::endl;}
    else{std::cout<<"qualcosa non va :("<<std::endl;}
}


{
std::cout<<"================ parallel_for incremento ================"<<std::endl;
    std::atomic<int> a=0;
    tp.parallel_for(start,end,[&](int i){
        a++;
    },[](int i){return ++i;});
    if(a.load() == n){std::cout<<" ok "<<std::endl;}
    else{std::cout<<"qualcosa non va :("<<std::endl;}
}   
    



    return 0; //
}
