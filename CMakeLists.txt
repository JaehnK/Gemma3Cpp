cmake_minimum_required(VERSION 3.14)
project(Gemma3Cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# llama.cpp 서브 프로젝트 추가
add_subdirectory(external/llama.cpp)

# 실행 파일 생성
add_executable(gemma3cpp
    src/main.cpp
    src/ModelManager.cpp
    src/EmbeddingGenerator.cpp
)

# 헤더 파일 경로 추가
target_include_directories(gemma3cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/llama.cpp/include
)

# llama 라이브러리 링크
target_link_libraries(gemma3cpp PRIVATE
    llama
)

# 컴파일 옵션
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gemma3cpp PRIVATE -Wall -Wextra)
endif()

