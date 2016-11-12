CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops

obj = sgd.o ftrl.o loss_func.o feature.o

sgd : $(obj)
	$(CXX) -g $(CFLAGS) -o sgd $(obj)

.PHONY : clean
clean : 
	rm sgd $(obj)

#%.o : %.cpp
#	$(CXX) -c $(CFLAGS) $< -O $@
