verifica che valori fittati in solver sequenziale e parallelo siano uguali.
(i GCV stesso lambda gia verificato in test normali)

compila il test_sanity(parallel) includendo sr_parallel_*.h e fe_ls_elliptic_parallel_*.h

g++-14  -I./../../../../fdaPDE-cpp -I./../../../../fdaPDE-cpp/fdaPDE/core -I/home/simo/eigen-3.4.0 -O2 -march=native -std=c++20 -s  test_sanitycheck_fitted.cpp -o test_sanitycheck_fitted

compila il test_sanity_seq includendo sr.h e fe_ls_elliptic.h

g++-14  -I./../../../../fdaPDE-cpp -I./../../../../fdaPDE-cpp/fdaPDE/core -I/home/simo/eigen-3.4.0 -O2 -march=native -std=c++20 -s  test_sanitycheck_fitted_seq.cpp -o test_sanitycheck_fitted_seq

eseguili entrambi che creano i file f_seq.txt ed f_parallel.txt

compila il test_verifica_uguali.cpp ed eseguilo che verifica i vettori nei txt siano uguali.  
g++-14  -I./../../../../fdaPDE-cpp -I./../../../../fdaPDE-cpp/fdaPDE/core -I/home/simo/eigen-3.4.0 -O2 -march=native -std=c++20 -s  test_verifica_uguali.cpp -o test_verifica_uguali