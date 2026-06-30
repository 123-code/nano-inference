// Minimal GGUF v3 reader + dequant for the minfer engine (M1).
// Header-only, no deps beyond libc/STL. mmaps the file and exposes tensors by
// name; dequantizes rows of Q4_0 / Q8_0 / Q6_K / F16 / F32 on demand (weights
// stay quantized in memory — a full f32 expansion of a 1.5B model is ~6 GB and
// won't fit this box).
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace minfer {

// --- ggml type ids we care about ---
enum GGMLType { GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q4_1 = 3, GT_Q8_0 = 8, GT_Q6_K = 14 };

struct TypeTraits { int block; size_t size; };  // elems/block, bytes/block
inline TypeTraits traits(uint32_t t) {
    switch (t) {
        case GT_F32:  return {1, 4};
        case GT_F16:  return {1, 2};
        case GT_Q4_0: return {32, 18};   // fp16 d + 16 packed bytes
        case GT_Q4_1: return {32, 20};   // fp16 d + fp16 m + 16 packed bytes
        case GT_Q8_0: return {32, 34};   // fp16 d + 32 int8
        case GT_Q6_K: return {256, 210}; // 128 ql + 64 qh + 16 sc + fp16 d
        default: fprintf(stderr, "minfer: unsupported ggml type %u\n", t); exit(1);
    }
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF, f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else { exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF; f = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

struct Tensor {
    uint32_t type = 0;
    std::vector<int64_t> ne;   // ne[0] = contiguous/row length
    const uint8_t* data = nullptr;
    int64_t rows() const { int64_t r = 1; for (size_t i = 1; i < ne.size(); i++) r *= ne[i]; return r; }
    int64_t cols() const { return ne[0]; }
    size_t nbytes() const { TypeTraits tr = traits(type); return (size_t)rows() * (ne[0] / tr.block) * tr.size; }
};

class GGUF {
public:
    explicit GGUF(const std::string& path) { load(path); }
    ~GGUF() { if (base_) munmap((void*)base_, fsize_); if (fd_ >= 0) close(fd_); }

    double meta_num(const std::string& k, double def) const {
        auto it = num_.find(k); return it == num_.end() ? def : it->second;
    }
    const std::string& meta_str(const std::string& k) const {
        static std::string empty; auto it = str_.find(k);
        return it == str_.end() ? empty : it->second;
    }
    const std::vector<std::string>& tokens() const { return tokens_; }

    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) { fprintf(stderr, "minfer: tensor not found: %s\n", name.c_str()); exit(1); }
        return it->second;
    }
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }

    // Dequantize row `r` of tensor `t` into out[0..cols-1].
    static void dequant_row(const Tensor& t, int64_t r, float* out) {
        const TypeTraits tr = traits(t.type);
        const int64_t cols = t.ne[0];
        const int64_t bpr = cols / tr.block;               // blocks per row
        const uint8_t* p = t.data + (size_t)(r * bpr) * tr.size;
        switch (t.type) {
            case GT_F32: memcpy(out, p, cols * 4); return;
            case GT_F16: { const uint16_t* h = (const uint16_t*)p;
                for (int64_t i = 0; i < cols; i++) out[i] = fp16_to_fp32(h[i]); return; }
            case GT_Q4_0:
                for (int64_t b = 0; b < bpr; b++, p += 18) {
                    float d = fp16_to_fp32(*(const uint16_t*)p);
                    const uint8_t* qs = p + 2; float* y = out + b * 32;
                    for (int j = 0; j < 16; j++) {
                        y[j]      = ((qs[j] & 0x0F) - 8) * d;
                        y[j + 16] = ((qs[j] >> 4)  - 8) * d;
                    }
                } return;
            case GT_Q4_1:
                for (int64_t b = 0; b < bpr; b++, p += 20) {
                    float d = fp16_to_fp32(*(const uint16_t*)p);
                    float m = fp16_to_fp32(*(const uint16_t*)(p + 2));
                    const uint8_t* qs = p + 4; float* y = out + b * 32;
                    for (int j = 0; j < 16; j++) {
                        y[j]      = (qs[j] & 0x0F) * d + m;
                        y[j + 16] = (qs[j] >> 4)  * d + m;
                    }
                } return;
            case GT_Q8_0:
                for (int64_t b = 0; b < bpr; b++, p += 34) {
                    float d = fp16_to_fp32(*(const uint16_t*)p);
                    const int8_t* qs = (const int8_t*)(p + 2); float* y = out + b * 32;
                    for (int j = 0; j < 32; j++) y[j] = qs[j] * d;
                } return;
            case GT_Q6_K:
                for (int64_t b = 0; b < bpr; b++, p += 210) {
                    const uint8_t* ql = p;
                    const uint8_t* qh = p + 128;
                    const int8_t*  sc = (const int8_t*)(p + 192);
                    float d = fp16_to_fp32(*(const uint16_t*)(p + 208));
                    float* y = out + b * 256;
                    for (int n = 0; n < 256; n += 128) {
                        for (int l = 0; l < 32; l++) {
                            int is = l / 16;
                            int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                            int8_t q3 = (int8_t)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                            int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                            y[l]      = d * sc[is + 0] * q1;
                            y[l + 32] = d * sc[is + 2] * q2;
                            y[l + 64] = d * sc[is + 4] * q3;
                            y[l + 96] = d * sc[is + 6] * q4;
                        }
                        ql += 64; qh += 32; sc += 8; y += 128;
                    }
                } return;
        }
    }

private:
    int fd_ = -1; size_t fsize_ = 0; const uint8_t* base_ = nullptr;
    const uint8_t* cur_ = nullptr;
    std::unordered_map<std::string, double> num_;
    std::unordered_map<std::string, std::string> str_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, Tensor> tensors_;

    template <typename T> T rd() { T v; memcpy(&v, cur_, sizeof(T)); cur_ += sizeof(T); return v; }
    std::string rd_str() { uint64_t n = rd<uint64_t>(); std::string s((const char*)cur_, n); cur_ += n; return s; }

    // read a metadata value of gguf type `vt`; store into num_/str_/tokens_ as relevant
    void rd_value(uint32_t vt, const std::string& key) {
        switch (vt) {
            case 0:  num_[key] = rd<uint8_t>();  break;
            case 1:  num_[key] = rd<int8_t>();   break;
            case 2:  num_[key] = rd<uint16_t>(); break;
            case 3:  num_[key] = rd<int16_t>();  break;
            case 4:  num_[key] = rd<uint32_t>(); break;
            case 5:  num_[key] = rd<int32_t>();  break;
            case 6:  num_[key] = rd<float>();    break;
            case 7:  num_[key] = rd<uint8_t>();  break;   // bool
            case 8:  str_[key] = rd_str();       break;
            case 10: num_[key] = (double)rd<uint64_t>(); break;
            case 11: num_[key] = (double)rd<int64_t>();  break;
            case 12: num_[key] = rd<double>();   break;
            case 9: {  // array
                uint32_t et = rd<uint32_t>(); uint64_t n = rd<uint64_t>();
                bool keep_tokens = (key == "tokenizer.ggml.tokens" && et == 8);
                if (keep_tokens) tokens_.reserve(n);
                for (uint64_t i = 0; i < n; i++) {
                    if (et == 8) { std::string s = rd_str(); if (keep_tokens) tokens_.push_back(std::move(s)); }
                    else rd_value(et, "");  // consume & discard
                } break;
            }
            default: fprintf(stderr, "minfer: bad meta type %u\n", vt); exit(1);
        }
    }

    void load(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { perror("open gguf"); exit(1); }
        struct stat st; fstat(fd_, &st); fsize_ = st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, fsize_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) { perror("mmap"); exit(1); }
        cur_ = base_;

        uint32_t magic = rd<uint32_t>();
        if (magic != 0x46554747u) { fprintf(stderr, "not a GGUF file\n"); exit(1); }
        uint32_t version = rd<uint32_t>(); (void)version;
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();

        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key = rd_str();
            uint32_t vt = rd<uint32_t>();
            rd_value(vt, key);
        }

        struct Info { Tensor t; uint64_t off; };
        std::vector<std::pair<std::string, Info>> infos;
        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name = rd_str();
            uint32_t nd = rd<uint32_t>();
            Tensor t; t.ne.resize(nd);
            for (uint32_t d = 0; d < nd; d++) t.ne[d] = (int64_t)rd<uint64_t>();
            t.type = rd<uint32_t>();
            uint64_t off = rd<uint64_t>();
            infos.push_back({name, {t, off}});
        }

        uint64_t align = (uint64_t)meta_num("general.alignment", 32);
        size_t pos = cur_ - base_;
        size_t data_start = (pos + align - 1) / align * align;
        for (auto& kv : infos) {
            kv.second.t.data = base_ + data_start + kv.second.off;
            tensors_.emplace(kv.first, kv.second.t);
        }
    }
};

}  // namespace minfer
