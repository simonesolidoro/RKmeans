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

#ifndef __SINGLETON_THREADPOOL_H__
#define __SINGLETON_THREADPOOL_H__

namespace fdapde {

class singleton_threadpool {
private:
    inline static bool status_ = false;
public:
    static threadpool<round_robin_scheduling,random_stealing>& instance() { 
        static threadpool<round_robin_scheduling,random_stealing> tp_(1024);
        return tp_;
    }

    static bool status() { return status_; }
    static void set_status_true() { status_ = true; }
    static void set_status_false() { status_ = false; }

};



}   // namespace fdapde

#endif 


/*

// utilizzatore
// in **qualunque** punto della codebase, posso accedere alla threadpool con

singleton::instance().parallel_for(...); // accesso alla threadpool del processo


// algoritmi in fdaPDE che hanno un implementazione parallela fanno

f(args..., execution::parallel);

nell'implementazione di f...

f(args..., execution::parallel) {
      singleton::instance().n_workers();
}



// -----------
opt.optimize(m.gcv(), execution::parallel);
// -------------- allo start di optimize(): singleton::set_status(); segnala l'intenzione di lavorare in parallelo
// -------------- alla fine farà clear_status(); ok, ho una threadpool, però la richiesta di esecuzione parallela
// ------------------- è finita

// nel gcv (ad esempio)
struct gcv {

      void double operator()(double lambda) {
            if(singleton::status()) {
                  // gcv sà che sta essendo eseguito in parallelo, quindi... agisce di conseguenza
            } else {
                  // esecuzione sequenziale
            }     
      }
      // ...
};
 */
//prova git