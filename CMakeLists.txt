project(pjtname)

# llama.cpp 불필요한 실행파일/테스트 끄기
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/llama.cpp)

include_directories(include)
file(GLOB SOURCES src/*.cpp)

add_executable(rag_demo ${SOURCES} main.cpp)
target_link_libraries(rag_demo PRIVATE llama)

