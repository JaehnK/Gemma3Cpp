#pragma once
#include <string>
#include <vector>
#include <memory>

#include "llama.h"

class EmbeddingGenerator {
    private:
        std::shared_ptr<ModelManager> modelMgr;
        llama_context* ctx;
        llama_context_params ctxParams;

        std::vector<llama_token> tokenize(const std::string& text);

    public:
        explicit EmbeddingGenerator(
            std::shared_ptr<ModelManager> ModelMgr,
            int n_ctx = 512,
            int n_batch = 512
        );
        // OCCF를 준수하지 않고 복사 연산자와 대입 연산자를 명시적으로 금지함
        // 이는 흠 ... 뭐냐, 그럴 수 있어 ...
        ModelManager(const ModelManager&) = delete;
        ModelManager& operator=(const ModelManager&) = delete;
        ~EmbeddingGenerator();

        std::vector<float> GenerateEmbedding(const std::string& txt);
        std::vector<float> generateNormalizedEmbedding(const std::string& text);
}