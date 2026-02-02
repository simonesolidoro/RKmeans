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

void pushbackconc(std::vector<int> V, fdapde::synchro_queue<int,fdapde::relaxed> & q){
    for (auto x: V){
        q.push_back(x);
        std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
    }
    if (q.empty()){std::cout<<"vuoto";}
}

int main(){
    int size_coda=30;
    fdapde::synchro_queue<int,fdapde::relaxed> q1(size_coda);
    
    std::vector<int> v;
    for (int i=0; i<size_coda; i++){
        v.push_back(i+1);
    }
    std::thread t1(pushbackconc,v,std::ref(q1));
    std::thread t2(pushbackconc,v,std::ref(q1));
    std::thread t3(pushbackconc,v,std::ref(q1));
    t1.join();
    t2.join();
    t3.join();
    q1.print();
    return 0;
}
