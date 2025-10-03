#include "../include/LLMCaller.hpp"
#include <stdexcept>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

// llama.cpp C API
#include "llama.h"

LLMCaller::LLMCaller(const std::string& model_path, const LLMParams& params)
: params_(params) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) throw std::runtime_error("LLMCaller: failed to load model: " + model_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params_.n_ctx;
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) throw std::runtime_error("LLMCaller: failed to create context");

    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) throw std::runtime_error("LLMCaller: failed to get vocab");
}

LLMCaller::~LLMCaller() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

std::string LLMCaller::token_to_piece(int token_id) const {
    std::string out;
    out.resize(8192);

    int32_t n = llama_token_to_piece(
        vocab_,
        token_id,
        &out[0],                // ✅ C++17 호환
        (int32_t)out.size(),
        /*special=*/false,
        /*lstrip=*/false
    );

    if (n < 0) throw std::runtime_error("llama_token_to_piece failed");
    out.resize(n);
    return out;
}

std::string LLMCaller::generate(const std::string& prompt) {
    if (!ctx_ || !model_) throw std::runtime_error("LLMCaller: context/model is null");

    // ✅ vocab_ 사용
    std::vector<llama_token> tokens(prompt.size() * 2);
    int n_tokens = llama_tokenize(
        vocab_,
        prompt.c_str(),
        (int)prompt.size(),
        tokens.data(),
        (int)tokens.size(),
        /*add_special=*/true,
        /*parse_special=*/false
    );
    if (n_tokens < 0) throw std::runtime_error("LLMCaller: tokenize failed");
    tokens.resize(n_tokens);

    // 2) 배치 준비
    llama_batch batch = llama_batch_init(
        std::max<int>(512, (int)tokens.size() + params_.max_tokens),
        /*embd=*/0,
        /*n_seq=*/1
    );

    // 3) 프롬프트 투입
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]   = tokens[i];
        batch.pos[i]     = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]  = false;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("LLMCaller: decode failed on prompt");
    }

    // 4) 생성 루프
    std::string out_text;
    const int eos_id = llama_vocab_eos(vocab_);
    int n_vocab = llama_vocab_n_tokens(vocab_);

    for (int step = 0; step < params_.max_tokens; ++step) {
        const float* logits = llama_get_logits(ctx_);
        if (!logits) break;

        // greedy argmax
        int best_id = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int id = 0; id < n_vocab; ++id) {
            float v = logits[id];
            if (v > best_logit) { best_logit = v; best_id = id; }
        }

        if (best_id == eos_id) break;

        out_text += token_to_piece(best_id);

        // 다음 step 배치 구성
        batch.n_tokens = 1;
        batch.token[0]   = best_id;
        batch.pos[0]     = n_tokens + step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]  = true;

        if (llama_decode(ctx_, batch) != 0) break;
    }

    llama_batch_free(batch);
    return out_text;
}

