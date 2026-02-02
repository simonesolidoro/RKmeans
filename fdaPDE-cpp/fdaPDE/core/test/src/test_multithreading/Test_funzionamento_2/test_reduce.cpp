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
    int size_v = std::stoi(argv[1]);
    fdapde::threadpool tp(1024,8);

{
std::cout<<"================ reduce sum  ================"<<std::endl; 
    double a = 2;
    std::vector<double> v (size_v,a);
    double sum = 0;
    sum = tp.reduce(v.begin(),v.end(),0.0,[](double a, double b){
        return a+b;
    }); 
    std::cout<<"atteso: "<<a*size_v<<" , ottenuto: "<< sum<<std::endl;
}
{
std::cout<<"================ reduce dot  ================"<<std::endl; 
    std::vector<double> v (size_v,1);
    double a = 13;
    v[0]=a;
    double dot = 0;
    dot = tp.reduce(v.begin(),v.end(),1.0,[](double a, double b){
        return a*b;
    }); 
    std::cout<<"atteso: "<<a<<" , ottenuto: "<< dot<<std::endl;
}
{
std::cout<<"================ reduce min  ================"<<std::endl; 
    double a = 1;
    std::vector<double> v (size_v,a+1);
    v[size_v/2]=a;
    double min = 0;
    min = tp.reduce(v.begin(),v.end(),a+2,[](double a, double b){
        return (a<=b)? a : b;
    }); 
    std::cout<<"atteso: "<<a<<" , ottenuto: "<< min<<std::endl;
}
{
std::cout<<"================ reduce max  ================"<<std::endl; 
    double a = 3;
    std::vector<double> v (size_v,a-1);
    v[size_v/2]=a;
    double max = 0;
    max = tp.reduce(v.begin(),v.end(),a-2,[](double a, double b){
        return (a>=b)? a : b;
    }); 
    std::cout<<"atteso: "<<a<<" , ottenuto: "<< max<<std::endl;
}


    return 0; //
}
