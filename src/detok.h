// GPT2 byte-level BPE detokenizer (shared by cuda_infer and mega).
#pragma once
#include <algorithm>
#include <string>
#include <vector>

// ---- GPT2 byte-level BPE detokenizer ----
// Qwen2 vocab pieces are stored as UTF-8 of a reversible byte->unicode map (the
// GPT2 "bytes_to_unicode" trick: control/space bytes are remapped to printable
// codepoints so the BPE alphabet has no whitespace). To get real text back we
// reverse it: decode each piece's UTF-8 to codepoints, map each codepoint to its
// original byte. cp2byte must be sized >= 324 (max remapped codepoint = 323).
static void build_byte_decoder(int* cp2byte) {
    for (int i = 0; i < 324; i++) cp2byte[i] = -1;
    std::vector<int> bs, cs;
    for (int b = '!';  b <= '~';  b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
    }
    for (size_t i = 0; i < bs.size(); i++) cp2byte[cs[i]] = bs[i];
}

// Decode one vocab piece (UTF-8 of remapped codepoints) into raw bytes, appended to out.
static void detok_piece(const std::string& s, const int* cp2byte, std::string& out) {
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = (unsigned char)s[i];
        int cp, len;
        if      (c < 0x80)        { cp = c;        len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; len = 4; }
        else { i++; continue; }
        for (int k = 1; k < len && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < 324 && cp2byte[cp] >= 0) out.push_back((char)cp2byte[cp]);
    }
}
