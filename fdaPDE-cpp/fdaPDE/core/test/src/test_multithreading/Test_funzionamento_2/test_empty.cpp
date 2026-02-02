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
 
 void vuoto(fdapde::synchro_queue<int,fdapde::relaxed> & q){
   std::cout<<std::this_thread::get_id()<<"dice che è: "<<q.empty()<<std::endl;
 }

 int main(){
{ std::cout<<"--------------------------------RELAXED--------------------------------"<<std::endl;
    fdapde::synchro_queue<int,fdapde::relaxed> q(10);
 
    std::vector<std::thread> pool;
    int k=0;
    for(int i=0; i<10; i++){
      if((k % 2) == 0)
         pool.emplace_back(&fdapde::synchro_queue<int,fdapde::relaxed>::push_front,std::ref(q),k);
      else
         pool.emplace_back(&fdapde::synchro_queue<int,fdapde::relaxed>::pop_front,std::ref(q));
      k++;
    }
    q.print();
    std::cout<<"vede: "<<q.empty()<<"   dovrebbe vedere 1 (coda vuota 1=true)"<<std::endl;
    q.print();
    
   

    for(size_t i =0; i<pool.size(); i++){
      pool[i].join();
    }
}

{/// deferred
  std::cout<<"--------------------------------deferred--------------------------------"<<std::endl;
  fdapde::synchro_queue<int,fdapde::deferred> q(10);
 
  std::vector<std::thread> pool;
  int k=0;
  for(int i=0; i<10; i++){
    if((k % 2) == 0)
       pool.emplace_back(&fdapde::synchro_queue<int,fdapde::deferred>::push_front,std::ref(q),k);
    else
       pool.emplace_back(&fdapde::synchro_queue<int,fdapde::deferred>::pop_front,std::ref(q));
    k++;
  }
  q.print();
  std::cout<<"vede: "<<q.empty()<<"   dovrebbe vedere 1 (coda vuota 1=true)"<<std::endl;
  q.print();
  
 

  for(size_t i =0; i<pool.size(); i++){
    pool[i].join();
  }
}
{/// blocking
  std::cout<<"--------------------------------blocking--------------------------------"<<std::endl;
  fdapde::synchro_queue<int,fdapde::blocking> q(10);
 
  std::vector<std::thread> pool;
  int k=0;
  for(int i=0; i<10; i++){
    if((k % 2) == 0)
       pool.emplace_back(&fdapde::synchro_queue<int,fdapde::blocking>::push_front,std::ref(q),k);
    else
       pool.emplace_back(&fdapde::synchro_queue<int,fdapde::blocking>::pop_front,std::ref(q));
    k++;
  }
  q.print();
  std::cout<<"vede: "<<q.empty()<<"   dovrebbe vedere 1 (coda vuota 1=true)"<<std::endl;
  q.print();

  for(size_t i =0; i<pool.size(); i++){
    pool[i].join();
  }
}


return 0;
    }
