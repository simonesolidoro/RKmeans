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
void printnum(){
    std::cout<<std::this_thread::get_id()<<std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
}
void printnum2(){
    std::cout<<std::this_thread::get_id()<<std::endl;
    std::this_thread::sleep_for(std::chrono::microseconds(2000));
}

bool fun(){
    std::cout<<"fun da thread_id: "<<std::this_thread::get_id()<<std::endl;
    return true;
};
bool fun2(){
    std::cout<<"fun22222222da thread_id: "<<std::this_thread::get_id()<<std::endl;
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

{
    using job = std::function<void()>;
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(64,16);
    std::vector<job> jobs;
    int push_count = 0;
    int n_jobs = 640;
    for(int i = 0; i< n_jobs; i++){
        if(i%2 == 0)
            jobs.push_back(printnum);
        else
            jobs.push_back(printnum2);
    }
    for (int i = 0; i<n_jobs; i++){
        tp.send(printnum);
    }
    

    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); //per aspettare esecuzione di tutti 
    std::cout<<push_count<<std::endl;
}
    return 0;
}
