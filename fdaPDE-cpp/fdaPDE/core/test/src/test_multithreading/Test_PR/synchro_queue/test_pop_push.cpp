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
    std::cout<<"size di code (10): ";
    int size;
    std::cin>>size;
{//relaxed_owait
    std::cout<<"---------------------------------test pop push con relaxed---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::relaxed> q(size);

    //push_front()
    for (int i =0; i<size; i++){
        q.push_front(i);
    }
    std::cout<<"push_front():  "<<std::endl;
    q.print();
    // std::cout<<"head: "<<q.get_head()<<std::endl;
    // std::cout<<"tail: "<<q.get_tail()<<std::endl;
    //pop_front
    for(int j=0; j<size; j++)
        q.pop_front();
    std::cout<<"pop_front():  "<<std::endl;
    q.print();

    //push_back()
    for (int i =0; i<size; i++){
        q.push_back(i);
    }
    std::cout<<"push_back():  "<<std::endl;
    q.print();
 
    //pop_back()
    for(int j=0; j<size; j++)
        q.pop_back();
    std::cout<<"pop_back():  "<<std::endl;
    q.print();
}



{//deferred
    std::cout<<"---------------------------------test pop push con deferred---------------------------------"<<std::endl;

    fdapde::synchro_queue<int,fdapde::deferred> q(size);

    //push_front()
    for (int i =0; i<size; i++){
        q.push_front(i);
    }
    std::cout<<"push_front():  "<<std::endl;
    q.print();
 
    //pop_front
    for(int j=0; j<size; j++)
        q.pop_front();
    std::cout<<"pop_front():  "<<std::endl;
    q.print();

    //push_back()
    for (int i =0; i<size; i++){
        q.push_back(i);
    }
    std::cout<<"push_back():  "<<std::endl;
    q.print();
 
    //pop_back()
    for(int j=0; j<size; j++)
        q.pop_back();
    std::cout<<"pop_back():  "<<std::endl;
    q.print();
}

{
    std::cout<<"---------------------------------test pop push con blocking---------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::blocking> q(size);

    //push_front()
    for (int i =0; i<size; i++){
        q.push_front(i);
    }
    std::cout<<"push_front():  "<<std::endl;
    q.print();
 
    //pop_front
    for(int j=0; j<size; j++)
        q.pop_front();
    std::cout<<"pop_front():  "<<std::endl;
    q.print();

    //push_back()
    for (int i =0; i<size; i++){
        q.push_back(i);
    }
    std::cout<<"push_back():  "<<std::endl;
    q.print();
 
    //pop_back()
    for(int j=0; j<size; j++)
        q.pop_back();
    std::cout<<"pop_back():  "<<std::endl;
    q.print();
}
     return 0;
    }
