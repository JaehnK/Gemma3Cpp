#pragma once
#include <string>
#include <memory>

#include "llama.h"

class ModelManager {
    private:
        llama_model* model;
        std::string modelPath;
    
    public:
        explicit ModelManager(const std::string& path);
        ModelManager(const ModelManager&) = delete;
        ModelManager& operator=(const ModelManager&) = delete;
        ~ModelManager();

        llama_model* getModel() const { return this->model; };
        std::string getModelPath() const { return this->modelPath; };

        int getEmbeddingDimension() const;
        int getVocabSize() const;
};