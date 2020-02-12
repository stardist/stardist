## Building the shared library libstardist3d (e.g. for use in Fiji)

**Work in progress!**

Currently tested only on linux and OSX

1. First install a recent version of gcc/g++, e.g. gcc-8 (not clang)

2. Build the library 

 `CC="gcc-8" CXX="g++-8" make`

3. Copy the library into the lib folder of Fiji 
 
 ```
 # osx
 cp libstardist3d.dylib $PATH_TO_FIJI/lib/macosx
 
 # linux
 cp libstardist3d.so $PATH_TO_FIJI/lib/linux64
 ```

4. Run the Stardist3D plugin and enjoy