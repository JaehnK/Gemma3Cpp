#pragma once
#include "llama.h"
#include <string>
#include <vector>

class EmbeddingModel {
    private:
        llama_model* model;
        llama_context* ctx;
        
    public:
        EmbeddingModel(const std::string& model_path) {
            llama_backend_init();
            
            auto model_params = llama_model_default_params();
            model = llama_load_model_from_file(model_path.c_str(), model_params);
            
            auto ctx_params = llama_context_default_params();
            ctx_params.embeddings = true;  // 임베딩 전용
            ctx_params.n_ctx = 512;
            ctx = llama_new_context_with_model(model, ctx_params);
        }
        
        std::vector<float> getEmbedding(const std::string& text) {
            // 토크나이징 및 임베딩 추출
            auto tokens = tokenize(text);
            auto batch = createBatch(tokens);
            
            llama_decode(ctx, batch);
            
            float* emb = llama_get_embeddings(ctx);
            int n_embd = llama_n_embd(model);
            
            std::vector<float> result(emb, emb + n_embd);
            llama_batch_free(batch);
            
            return result;
        }
        
        ~EmbeddingModel() {
            llama_free(ctx);
            llama_free_model(model);
        }
    };