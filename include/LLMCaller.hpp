#pragma once
#include <string>
#include <cstddef>

#include "llama.h"

struct llama_model;
struct llama_context;

struct LLMParams {
    int   n_ctx        = 2048;
    int   max_tokens   = 128;
    float temperature  = 0.0f; // 0=greedy. 샘플링은 추후 필요 시 추가
};

class LLMCaller {
public:
    explicit LLMCaller(const std::string& model_path, const LLMParams& params = {});
    ~LLMCaller();

    // prompt -> generated text
    std::string generate(const std::string& prompt);

    // 복사 금지, 이동 허용
    LLMCaller(const LLMCaller&) = delete;
    LLMCaller& operator=(const LLMCaller&) = delete;
    LLMCaller(LLMCaller&&) = delete;
    LLMCaller& operator=(LLMCaller&&) = delete;

private:
    const llama_vocab * vocab_ = nullptr;
    llama_model*  model_ = nullptr;
    llama_context* ctx_  = nullptr;
    LLMParams      params_{};

    // 내부 유틸
    std::string token_to_piece(int token_id) const;
};