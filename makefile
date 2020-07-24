APPNAME=demo
OBJS=mr-test.o MetaRecognition.o weibull.o weibull_gpu.o GpuSolver.o 
CXX=g++ -w -m64 -std=c++11
CXXFLAGS = -Wall -O3 

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61  
NVCC=nvcc

default: $(APPNAME)

demo: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o