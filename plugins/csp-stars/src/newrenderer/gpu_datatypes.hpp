#pragma once

#include <cstdint>

//#pragma pack(push, 1)

struct GpuChunkMeta {
    uint32_t startRowNum = 0; //where in batch memory does the first batch start; (1 step = 256 x 256 x 4 bytes)
    uint32_t rowCount = 0; //how many rows in this chunk
    uint32_t pad0;//lastBatchRowCount = 0; //leftover point rows in last batch
    uint32_t chunkID = 0; //unique location bound id, multiple batches can be assigned to one id
    //if we pack batchCount or lastRowCount, maybe we could use the leftover space to encode some LOD helper info or smth
};

//IDEA: overloaded function "gpu-serialize" that defines how to lay out data vector before passing to SSBO upload 

//#pragma pack(pop)

constexpr size_t GPU_THREAD_COUNT = 256; //how many parallel columns per batch

//for format where first uint32_t encodes position and second uint32_t encodes lumi-temp (interleaved)
using GpuBatchRow = std::array<uint32_t, GPU_THREAD_COUNT * 2>;