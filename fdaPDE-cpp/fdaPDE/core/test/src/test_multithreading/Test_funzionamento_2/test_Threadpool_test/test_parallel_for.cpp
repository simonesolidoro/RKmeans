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
{   
    int granularity = std::stoi(argv[1]);
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(2048,8);
    
{// parallel_for  gran1 int
    int end = 1001;
    int start = 10;
    std::vector<int> seq;
    for(size_t i = start; i<end; i++){
        seq.push_back(i);
    } 
    std::vector<int> par(end-start);
    tp.parallel_for_merge(start,end,[&](int i){
        par[i-start] = i;
    });

    int uguali = seq.size() == par.size();
    for(int i = 0; i<seq.size(); i++){
        if(seq[i] != par[i]){
            uguali *= 0;
        } 
    }
    if(!uguali){std::cout<<"gran1 int NON funziona";}
    else{std::cout<<"gran1 int funziona";}
}
std::cout<<std::endl; 
{// parallel_for  gran1 it
    int end = 10000;
    int start = 0;
    std::vector<int> seq;
    for(size_t i = start; i<end; i++){
        seq.push_back(i);
    } 
    std::vector<int> par(end-start);
    tp.parallel_for(par.begin(),par.end(),[&](std::vector<int>::iterator i){
        *i = (i-par.cbegin());
    });

    int uguali = seq.size() == par.size();
    for(int i = 0; i<seq.size(); i++){
        if(seq[i] != par[i]){
            uguali *= 0;
        } 
    }
    if(!uguali){std::cout<<"gran1 iterator NON funziona";}
    else{std::cout<<"gran1 iterator funziona";}
} 
std::cout<<std::endl; 
{// parallel_for  gran_input int
    int end = 10000;
    int start = 0;
    std::vector<int> seq;
    for(size_t i = start; i<end; i++){
        seq.push_back(i);
    } 
    std::vector<int> par(end-start);
    tp.parallel_for(start,end,[&](int i, int worker_index){
        par[i-start] = i;
    },granularity);

    int uguali = seq.size() == par.size();
    for(int i = 0; i<seq.size(); i++){
        if(seq[i] != par[i]){
            uguali *= 0;
        } 
    }
    if(!uguali){std::cout<<"gran_input int NON funziona";}
    else{std::cout<<"gran_input int funziona";}
}
std::cout<<std::endl; 
{// parallel_for  gran1 it
    int end = 10000;
    int start = 0;
    std::vector<int> seq;
    for(size_t i = start; i<end; i++){
        seq.push_back(i);
    } 
    std::vector<int> par(end-start);
    tp.parallel_for(par.begin(),par.end(),[&](std::vector<int>::iterator i, int worker_index){
        *i = (i-par.cbegin());
    },granularity);

    int uguali = seq.size() == par.size();
    for(int i = 0; i<seq.size(); i++){
        if(seq[i] != par[i]){
            uguali *= 0;
        } 
    }
    if(!uguali){std::cout<<"gran_input iterator NON funziona";}
    else{std::cout<<"gran_input iterator funziona";}
}
std::cout<<std::endl; 


return 0;
}
