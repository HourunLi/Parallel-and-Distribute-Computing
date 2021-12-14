main: main.cc
	g++ main.cc -o main -Ofast -fopenmp -funroll-loops -mprefer-vector-width=512 -fopt-info-vec-optimized -march=cascadelake -mcmodel=large
clean:
	rm -f main
