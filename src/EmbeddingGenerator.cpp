#include "../include/EmbeddingGenerator.hpp"

#include <stdexcept>
#include <cmath>
#include <algorithm>

EmbeddingGenerator::EmbeddingGenerator(
    std::shared_ptr<ModelManager> modelMgr,
    int n_ctx,
    int n_batch)
    : modelMgr(std::move(modelMgr)), ctx(nullptr) {

    if (!modelMgr || !modelMgr->getModel())
        throw std::invalid_argument("Invalid ModelManager");

    // 컨텍스트 파라미터 설정
    ctxParams = llama_context_default_params();
    ctxParams.n_ctx = n_ctx;
    ctxParams.n_batch = n_batch;
    ctxParams.embeddings = true;
    ctxParams.n_threads = 4;

    // 컨텍스트 생성
    ctx = llama_new_context_with_model(this->modelMgr->getModel(), ctxParams);

    if (!ctx)
        throw std::runtime_error("Failed to create context");
}

EmbeddingGenerator::~EmbeddingGenerator() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    // model_manager는 shared_ptr이므로 자동으로 관리됨
}

std::vector<llama_token> EmbeddingGenerator::tokenize(const std::string& text, bool add_bos) {
    
    std::vector<llama_token> tokens(text.length() + (add_bos ? 1 : 0));
    int n_tokens = llama_tokenize(
        this->modelMgr->getModel(),
        text.c_str(),
        text.length(),
        tokens.data(),
        (int)tokens.size(),
        add_bos,
        false
    );
    
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            this->modelMgr->getModel(),
            text.c_str(),
            text.length(),
            tokens.data(),
            (int)tokens.size(),
            add_bos,
            false
        );
    }
    
    tokens.resize(n_tokens);
    return tokens;
}

std::vector<float> EmbeddingGenerator::generateEmbedding(
    const std::string& text) {
    
    if (text.empty()) {
        throw std::invalid_argument("Input text is empty");
    }
    
    auto tokens = tokenize(text, true);
    
    if (tokens.empty()) {
        throw std::runtime_error("Tokenization failed");
    }
    
    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], (int)i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to decode");
    }
    
    int n_embd = this->modelMgr->getEmbeddingDimension();
    const float* embeddings = llama_get_embeddings(ctx);
    
    if (!embeddings) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to get embeddings");
    }
    
    std::vector<float> result(embeddings, embeddings + n_embd);
    llama_batch_free(batch);
    
    return result;
}

std::vector<float> EmbeddingGenerator::generateNormalizedEmbedding(
    const std::string& text) {
    
    auto embedding = generateEmbedding(text);
    
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0.0f) {
        for (float& val : embedding) {
            val /= norm;
        }
    }
    
    return embedding;
}

int EmbeddingGenerator::getEmbeddingDimension() const {
    return this->modelMgr->getEmbeddingDimension();
}