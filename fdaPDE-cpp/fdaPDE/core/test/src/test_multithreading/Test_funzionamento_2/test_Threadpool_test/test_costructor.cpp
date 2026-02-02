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
struct A{
    int a;
};
A incrementaA(A& aa){
    aa.a++;
    return aa;
}
void printnum(){
    std::cout<<std::this_thread::get_id()<<std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(10000));
}
bool fun(){
    std::cout<<"fun da thread_id: "<<std::this_thread::get_id()<<std::endl;
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return true;
};
bool fun2(){
    std::cout<<"fun2da thread_id: "<<std::this_thread::get_id()<<std::endl;
    return true;
};

bool count (int n){
    int i= 0;
    for(int j=0; j<n; j++){
        i++;
    }
    std::cout<<"thread:"<<std::this_thread::get_id()<<" ha contato fino a "<<i<<std::endl;
    return true;
};
int main(){
    using job = std::function<void()>;
    fdapde::threadpool<fdapde::round_robin_scheduling ,fdapde::random_stealing> tp;
    fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::max_load_stealing> tp2(100);
{
    std::vector<std::optional<std::future<void>>> futs;
    std::vector<job> jobs;
    int n_jobs = 64;
    for (int i = 0; i<n_jobs; i++){
        futs.push_back(std::move(tp.send(printnum)));
    }
    for (int i = 0; i<n_jobs; i++){
        if(futs[i]){
            futs[i].value().get();}
        else 
            std::cout<<"noninviato"<<std::endl;
    }
}
{
    std::vector<std::optional<std::future<void>>> futs;
    std::vector<job> jobs;
    int n_jobs = 64;
    for (int i = 0; i<n_jobs; i++){
        futs.push_back(std::move(tp2.send(printnum)));
    }
    for (int i = 0; i<n_jobs; i++){
        if(futs[i]){
            futs[i].value().get();}
        else 
            std::cout<<"noninviato"<<std::endl;
    }
}
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); //per aspettare esecuzione di tutti 
    return 0;
}
