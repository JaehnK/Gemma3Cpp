#pragma once
#include <string>
#include <vector>

class Retriever {
public:
    Retriever();
    ~Retriever();

    std::vector<std::string> retrieve(const std::string& query);
};
