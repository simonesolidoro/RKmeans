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

    fdapde::Synchro_queue<value,fdapde::hold_wait> q1(size_coda);
    value el = "ciao";

//pop_back() singolo thread

        //popolo
        for (int i=0; i<size_coda; i++){
            q1.push_front(el);
        }
    auto start7 = std::chrono::high_resolution_clock::now();
    for(int j=0; j<size_coda-1; j++){
        q1.pop_back();
    }

    auto end7 = std::chrono::high_resolution_clock::now();
    auto duration7 = std::chrono::duration_cast<std::chrono::microseconds>(end7 - start7);  
    //std::cout<<"pop_back() worker_queue di n_elementi: "<<size_coda<<" impiegato:"<<duration7.count()<< " microsecondi\n";
    std::cout<<duration7.count()<<",";
    return 0;
}