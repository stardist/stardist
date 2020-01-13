building the shared library libstardist3d.dylib (e.g. for use in Fiji)



# OSX (use gcc, not clang)

mkdir -p build; 
cd build;
gcc-8 -c  -O3 -I../qhull_src/src/ ../qhull_src/src/libqhull_r/*.c
g++-8 -c -O3 -I../qhull_src/src/ ../qhull_src/src/libqhullcpp/*.cpp 
cd -

g++-8 -dynamiclib -std=c++11 -O3 -fopenmp -I./qhull_src/src/ -o libstardist3d.dylib stardist3d_impl.cpp stardist3d_lib.c stardist3d_lib.h utils.cpp build/*.o 


# linux

mkdir -p build; 
cd build;
g++-8 -c -O3 -fPIC -fpermissive -I../qhull_src/src/ ../qhull_src/src/libqhull_r/*.c
g++-8 -c -O3 -fPIC -I../qhull_src/src/ ../qhull_src/src/libqhullcpp/*.cpp 
cd -

g++-8 -shared -fPIC -std=c++11 -O3 -fopenmp -I./qhull_src/src/ -o libstardist3d.so stardist3d_impl.cpp stardist3d_lib.c stardist3d_lib.h utils.cpp build/*.o 






