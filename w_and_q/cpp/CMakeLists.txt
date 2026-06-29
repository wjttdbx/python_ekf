cmake_minimum_required(VERSION 3.16)
project(jt_ekf_attitude LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 主程序
add_executable(jt_ekf_sim main.cpp)

# 包含头文件
target_include_directories(jt_ekf_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# 优化选项
target_compile_options(jt_ekf_sim PRIVATE -O2 -Wall -Wextra)
