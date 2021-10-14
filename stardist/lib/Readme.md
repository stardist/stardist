## Building the shared library libstardist3d (e.g. for use in Fiji)

**Work in progress!**

Currently tested only on linux and OSX

* First install a recent version of gcc/g++, e.g. gcc-11 (not clang)

* Build the library 


 ```
 cd stardist/stardist/lib
 CC="gcc-11" CXX="g++-11" make
 ```


## Usage 

### From C++ 

* See `test_lib3d.cpp` for a simple example of how to use non maximum suppression (NMS) from c++

* build example with `make test` 

* run with `./test_lib3d`



### Fiji (experimental, not supported yet)

* Copy the library into the lib folder of Fiji 
 
 ```
 # osx
 cp libstardist3d.dylib $PATH_TO_FIJI/lib/macosx
 
 # linux
 cp libstardist3d.so $PATH_TO_FIJI/lib/linux64
 ```

* Run the Stardist3D plugin and enjoy