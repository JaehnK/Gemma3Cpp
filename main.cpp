#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Retriever.hpp"
#include "RAG.hpp"
#include "LLMCaller.hpp"

int main() {
    // --- 1) 모델 경로 하드코딩 ---
    const std::string model_path = "./models/gemma-3-270m-it-Q4_K_S.gguf";

    // --- 2) LLM 설정 ---
    LLMParams params;
    params.n_ctx = 512;
    params.max_tokens = 128;   // 생성할 최대 토큰 수

    try {
        // --- 3) Retriever/LLMCaller/RAG 초기화 ---
        auto retriever = std::make_unique<Retriever>();
        auto llm = std::make_unique<LLMCaller>(model_path, params);
        RAG rag(std::move(retriever), std::move(llm));

        // --- 4) 시작 정보 표시 ---
        std::cout << "=== RAG Demo ===\n";
        std::cout << "Model loaded from: " << model_path << "\n";
        std::cout << "Context size: " << params.n_ctx
                  << ", Max tokens: " << params.max_tokens << "\n";
        std::cout << "Type a question (or 'exit' to quit)\n\n";

        // --- 5) 반복 입력 루프 ---
        std::string query;
        while (true) {
            std::cout << "> ";
            if (!std::getline(std::cin, query)) break;
            if (query == "exit") break;
            if (query.empty()) continue;

            // 5-1) 사용자 입력 로그
            std::cout << "[User Query] " << query << "\n";

            // 5-2) RAG ask 호출
            std::string answer = rag.ask(query);

            // 5-3) 결과 출력
            std::cout << "[Answer] " << answer << "\n\n";
        }

        std::cout << "Exiting RAG demo.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
