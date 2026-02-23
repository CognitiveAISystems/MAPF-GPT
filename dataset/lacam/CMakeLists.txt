cmake_minimum_required(VERSION 3.16)
project(lacam-project CXX)

add_subdirectory(./lacam3)

add_library(lacam SHARED main.cpp)
target_compile_features(lacam PUBLIC cxx_std_17)
target_include_directories(lacam PRIVATE ./lacam3/include)
target_link_libraries(lacam lacam3 libpthread.so)