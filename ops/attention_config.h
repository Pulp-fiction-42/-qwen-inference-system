#pragma once

// 从原 layer.h 搬入的 GQA 参数，仅供算子内部使用。
// 不包含模型类型；layer_kv 是原结构体中未被 kernel 读取的字段。
struct GQAConfig {
    int seq_len;
    int head_dim;
    int max_seq_len;
    float* layer_kv;
    int heads_num;
    int kv_head_dim;
};
