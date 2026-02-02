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

void printmatrix(std::vector<std::vector<int>> M, int size){
    for (int i = 0; i<size; i++){
        for (int j = 0; j<size; j++){
            std::cout<<M[i][j]<<" ";    
        }
        std::cout<<std::endl;
    }
}

int main(int argc, char** argv){
    int size = std::stoi(argv[1]); //numero cicli in contafino // lunghezza singolo job
    std::vector<std::vector<int>> A;
    std::vector<std::vector<int>> B;
    std::vector<std::vector<int>> C;

    for (int i = 0; i<size; i++){
        A.emplace_back(size,1);
        B.emplace_back(size,1);
        C.emplace_back(size,0);
    }
   
    auto start2 = std::chrono::high_resolution_clock::now();

    for(int i = 0; i<size; i++){
        for(int j=0; j<size; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    
/*   
  // copia di colonna ad ogni iterazione 
    for(int i = 0; i<size; i++){
        std::vector<int> A_col_i (A[i]);
        std::vector<int> B_col_i (B[i]);
        for(int j=0; j<size; j++){
            C[i][j] = A_col_i[j] + B_col_i[j];
        }
    }
 */

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2); 
    //printmatrix(C,size);
    std::cout<<duration2.count()<<",";
    return 0;
}   