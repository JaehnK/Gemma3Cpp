#include "../include/ModelManager.hpp"
#include <stdexcept>

ModelManager::ModelManager(const std::string& path)
    : model(nullptr), modelPath(path) {

    // llama 초기화
    llama_backend_init();

    // 모델의 파라미터 설정
    llama_model_params modelParams = llama_model_default_params();

    model = llama_model_load_from_file(path.c_str(), modelParams);
    if (!model) {
        llama_backend_free();
        throw std::runtime_error("Failed to load model from: " + path);
    }
}

ModelManager::~ModelManager() {
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }

    llama_backend_free();
}

int ModelManager::getEmbeddingDimension() const {
    if (!model)
        throw std::runtime_error("Model Not Loaded");

    return llama_model_n_embd(model);
}

int ModelManager::getVocabSize() const {
    if (!model)
        throw std::runtime_error("Model Not Loaded");

    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}


