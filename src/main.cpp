#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "../include/ModelManager.hpp"
#include "../include/EmbeddingGenerator.hpp"

int main(int argc, char** argv) {
    // 기본 모델 경로 설정
    const std::string defaultModelPath = "models/gemma-3-270m/gemma-3-270m-Q8_0.gguf";
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <text> [model_path]\n";
        std::cerr << "Default model: " << defaultModelPath << "\n";
        return 1;
    }

    std::string inputText;
    std::string modelPath;
    
    // 인자가 2개면: <text>만 제공 (기본 모델 사용)
    // 인자가 3개 이상이면: 마지막 인자를 모델 경로로 사용
    if (argc == 2) {
        inputText = argv[1];
        modelPath = defaultModelPath;
    } else {
        // 마지막 인자 전까지를 텍스트로 결합
        for (int i = 1; i < argc - 1; ++i) {
            if (i > 1) inputText += ' ';
            inputText += argv[i];
        }
        modelPath = argv[argc - 1];
    }

    try {
        auto modelMgr = std::make_shared<ModelManager>(modelPath);
        EmbeddingGenerator generator(modelMgr, 512, 512);

        std::vector<float> embedding = generator.generateEmbedding(inputText);

        std::cout << "Embedding dimension: " << embedding.size() << "\n";
        std::cout << "First values: ";
        const size_t preview = std::min<size_t>(embedding.size(), 8);
        for (size_t i = 0; i < preview; ++i) {
            if (i) std::cout << ", ";
            std::cout << embedding[i];
        }
        std::cout << "\n";

        std::vector<float> normalized = generator.generateNormalizedEmbedding(inputText);
        float sumSquares = 0.0f;
        for (float v : normalized) sumSquares += v * v;
        std::cout << "Normalized L2 norm^2: " << sumSquares << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}

