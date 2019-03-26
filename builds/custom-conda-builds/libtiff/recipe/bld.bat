mkdir build_%c_compiler%
cd build_%c_compiler%

cmake -G"%CMAKE_GENERATOR%"                      ^
      -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%"  ^
      -DCMAKE_BUILD_TYPE=Release                 ^
      -DCMAKE_C_FLAGS="%CFLAGS% -DWIN32"         ^
      -DCMAKE_CXX_FLAGS="%CXXFLAGS% -EHsc"       ^
      ..
if errorlevel 1 exit /b 1

cmake --build . --target install --config Release
if errorlevel 1 exit /b 1
