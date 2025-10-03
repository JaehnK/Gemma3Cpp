#include "../include/Retriever.hpp"

Retriever::Retriever() = default;
Retriever::~Retriever() = default;

std::vector<std::string> Retriever::retrieve(const std::string& query) {
    // TODO: 실제 vector DB 붙일 때 교체
    return { "Dummy doc 1 for: " + query,
             "Dummy doc 2 for: " + query };
}
