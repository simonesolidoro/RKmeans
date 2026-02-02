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

void contafino(int n){
    int a= 0;
    for (int j = 0; j<n; j++){
        a++;
        a--;
        a++;
        a--;
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));

}
int main(int argc, char** argv){
    int n = std::stoi(argv[1]); //numero cicli in contafino

    int n_thread = std::stoi(argv[2]);
    int n_job = 100;
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::max_load_stealing> tp(n_job,n_thread);
    std::atomic<int> a = 0;

    

    std::vector<std::function<void(int)>> jobs;
    std::vector<std::optional<std::future<void>>> futs;
    for(int i= 0; i<n_job; i++){
        jobs.push_back(contafino);
    }
    
    auto start2 = std::chrono::high_resolution_clock::now();

    for(int i= 0; i<n_job; i++){
        futs.push_back(std::move(tp.send(contafino,n)));
    }
    for(int i= 0; i<n_job; i++){
        if(futs[i]){
            futs[i].value().get();
        }
    }

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    std::cout<<"operazioni fatte: "<<n*n_job<<"  con n_job: "<<n_job<<"mandati su n_thread:"<<n_thread<<" impiegato:"<<duration2.count()<< " microsecondi\n";
    
    auto start3 = std::chrono::high_resolution_clock::now();
    std::thread t(contafino,n*n_job);
    t.join();
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    std::cout<<"operazioni fatte: "<<n*n_job<<" singolo thread impiegato:"<<duration3.count()<< " microsecondi\n";
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    int T = std::thread::hardware_concurrency();
    std::cout<<"hardware thread supportati: "<<T<<std::endl;
    return 0;
}


