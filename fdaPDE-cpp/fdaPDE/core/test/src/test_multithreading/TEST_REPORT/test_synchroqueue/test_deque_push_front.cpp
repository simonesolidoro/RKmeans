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


using value = std::string; 


int main(int argc, char** argv){
    int size_coda= std::stoi(argv[1]);

    value el = "ciao";

    Worker_queue_deque<value> q1;

//push_front() singolo thread

    auto start = std::chrono::high_resolution_clock::now();
    for(int j=0; j<size_coda-1; j++){
        q1.push_front(el);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    //std::cout<<"push_frot deque di n_elementi: "<<size_coda<<" impiegato:"<<duration.count()<< " microsecondi\n";
    std::cout<<duration.count()<<",";
    return 0;
}
