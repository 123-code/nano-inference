// minfer M1 — reference oracle built on the prebuilt libllama.
// Tokenizes a prompt and greedily decodes N tokens on CPU, printing token IDs
// in a machine-parseable form so cpu_infer can be checked against it.
//
//   ref_oracle <model.gguf> <n_gen> "<prompt>"
//   INPUT <id...>
//   OUTPUT <id...>
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_gen> <prompt>\n", argv[0]); return 1; }
    const char* path = argv[1];
    int n_gen = atoi(argv[2]);
    std::string prompt = argv[3];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU: deterministic, no GPU-memory contention
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024;
    llama_context* ctx = llama_init_from_model(model, cp);

    // tokenize (let the model decide special tokens; parse <|im_start|> etc.)
    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, true, true);

    printf("INPUT");
    for (auto t : toks) printf(" %d", t);
    printf("\n");

    llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode prompt failed\n"); return 1; }

    int nvocab = llama_vocab_n_tokens(vocab);
    printf("OUTPUT");
    llama_token cur;
    for (int i = 0; i < n_gen; i++) {
        float* logits = llama_get_logits_ith(ctx, -1);
        cur = 0; float best = logits[0];
        for (int v = 1; v < nvocab; v++) if (logits[v] > best) { best = logits[v]; cur = v; }
        printf(" %d", cur);
        batch = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode step failed\n"); return 1; }
    }
    printf("\n");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
