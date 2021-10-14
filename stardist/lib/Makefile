UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	LIB=libstardist3d.so
	CFLAGSQHULL=-fPIC -O3
	CFLAGSLIB=-shared -fPIC
else ifeq ($(UNAME), Darwin)
	LIB=libstardist3d.dylib
	CFLAGSQHULL=-O3 
	CFLAGSLIB=-dynamiclib 
endif

CFLAGS=-std=c++11 -O3 -fopenmp
BUILDDIR=build


all: lib

qhull_cpp:
	mkdir -p $(BUILDDIR) && cd $(BUILDDIR) && $(CXX) -c $(CFLAGSQHULL) -fpermissive -I../external/qhull_src/src/ ../external/qhull_src/src/libqhullcpp/*.cpp 

qhull_c:
	mkdir -p $(BUILDDIR) && cd $(BUILDDIR) && $(CC) -c $(CFLAGSQHULL) -I../external/qhull_src/src/ ../external/qhull_src/src/libqhull_r/*.c

lib: qhull_c qhull_cpp stardist3d_impl.cpp stardist3d_lib.c stardist3d_lib.h utils.cpp 
	$(CXX) $(CFLAGSLIB) $(CFLAGS) -o $(LIB) stardist3d_impl.cpp stardist3d_lib.c stardist3d_lib.h utils.cpp -I./external/qhull_src/src/ -I./external/nanoflann $(BUILDDIR)/*.o 



test: test_lib3d.cpp
	$(CXX) test_lib3d.cpp -o test_lib3d -lstardist3d -L.
