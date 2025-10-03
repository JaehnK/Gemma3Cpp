#pragma once
#include <string>
#include <memory>
#include "Retriever.hpp"
#include "LLMCaller.hpp"

class RAG {
public:
    RAG(std::unique_ptr<Retriever> retriever,
        std::unique_ptr<LLMCaller> llm);
    ~RAG();

    std::string ask(const std::string& query);

private:
    std::unique_ptr<Retriever> retriever_;
    std::unique_ptr<LLMCaller> llm_;
};