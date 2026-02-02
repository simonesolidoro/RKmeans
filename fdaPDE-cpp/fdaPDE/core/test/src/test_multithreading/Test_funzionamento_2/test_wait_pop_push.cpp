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

   
int main(){


// blocking
{
    std::cout<<"---------------------------------test push back/front wait_for con blocking---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::blocking> q1(5);
    //popolo
    for(int j=0; j<5; j++){
        q1.push_front(j);
    }
    //push
    std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::push_back_or_wait_for,std::ref(q1),9,5); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
    q1.pop_back();
    q1.print(); 
    q1.print();
    d.join();

    std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::push_front_or_wait_for,std::ref(q1),9,5); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
    q1.pop_front();
    q1.print(); 
    d2.join();
    q1.print();

}
{
    std::cout<<"---------------------------------test pop bback/front wait_for con blocking---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::blocking> q1(5);

    //pop
    std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::pop_back_or_wait_for,std::ref(q1),5); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
    q1.push_back(9);
    q1.print(); 
    q1.print();
    d.join();

    std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::pop_front_or_wait_for,std::ref(q1),5); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
    q1.push_front(9);
    q1.print(); 
    d2.join();
    q1.print();

}
{
    std::cout<<"---------------------------------test push back/front wait con blocking---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::blocking> q1(5);
    //popolo
    for(int j=0; j<5; j++){
        q1.push_front(j);
    }
    //push
    std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::push_back_or_wait,std::ref(q1),9); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
    q1.pop_back();
    q1.print(); 
    q1.print();
    d.join();

    std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::push_front_or_wait,std::ref(q1),9); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
    q1.pop_front();
    q1.print(); 
    d2.join();
    q1.print();

}
{
    std::cout<<"---------------------------------test pop bback/front wait con blocking---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::blocking> q1(5);

    //pop
    std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::pop_back_or_wait,std::ref(q1)); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
    q1.push_back(9);
    q1.print(); 
    q1.print();
    d.join();

    std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::pop_front_or_wait,std::ref(q1)); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
    q1.push_front(9);
    q1.print(); 
    d2.join();
    q1.print();

}
return 0;
}

