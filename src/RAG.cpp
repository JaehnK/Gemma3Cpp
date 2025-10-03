#include "../include/RAG.hpp"

RAG::RAG(std::unique_ptr<Retriever> retriever,
         std::unique_ptr<LLMCaller> llm)
: retriever_(std::move(retriever)), llm_(std::move(llm)) {}

RAG::~RAG() = default;

std::string RAG::ask(const std::string& query) {
    // 1) 문서 집계
    const auto docs = retriever_->retrieve(query);

    // 2) 간단 컨텍스트 결합 (추후 포맷/프롬프트 템플릿 교체)
    std::string context;
    context.reserve(4096);
    for (const auto& d : docs) {
        context.append(d);
        context.push_back('\n');
    }

    const std::string final_prompt =
        "You are a helpful assistant.\n"
        "Use the following context to answer the question.\n\n"
        "Context:\n" + context +
        "\nQuestion:\n" + query +
        "\nAnswer:";

    // 3) LLM 호출
    return llm_->generate(final_prompt);
}