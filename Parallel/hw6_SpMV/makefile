spmv: main.cu spmv_util.cc
	g++  spmv_util.cc -c -o spmv_util.o
	nvcc main.cu -c -o main.o
	nvcc main.o spmv_util.o  -o spmv
	rm *.o
clean:
	rm *.o
	rm spmv