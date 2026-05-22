#include "StarRenderer.hpp"

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "util/shader_utils.hpp"
#include "util/util.hpp"
#include "util/gpu_stopwatch.hpp"

#include "datatypes.hpp"
#include "gpu_datatypes.hpp"


constexpr size_t GPU_TILE_SIZE_PX = 64;
constexpr size_t GPU_MAX_TILE_CHUNKS = 15;

namespace {

GLuint createTextureR32F(glm::uvec2 size) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size.x, size.y, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

GLuint createTextureRGBA8(glm::uvec2 size) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

}

/*double pixel_world_space_width(double dist) {
    return 2.0 * dist / (960.0); //tan(45°) = 1
}*/ //TODO: use this to generate dynamic LOD levels, for now we assume fixed precision

//TODO fix chunking logic, changing this parameter just scales everything
constexpr float CHUNK_SIZE_PARSECS = 12000.0 / 16; // 64^3 chunks, spanning 12^3 kpc^3, 12000 / 64 = 187.5

using RawStar = StarRenderer::RawStar;

void bin_stars_to_chunks(std::vector<RawStar> const& raw_stars, std::unordered_map<uint32_t, Chunk>& chunksOut) {
    std::cout << "global position of first raw star: " << glm::to_string(raw_stars.front().mPosition) << std::endl;
    std::cout << "number of input raw rawStars:" << raw_stars.size() << std::endl;

    //store to internal chunks first, copy out later while merging overlapping rawStars
    std::unordered_map<uint32_t, Chunk> chunksInternal;

    for(RawStar const& raw_star : raw_stars) {
        glm::vec3 const globalChunkSpacePosition = raw_star.mPosition / CHUNK_SIZE_PARSECS;
        glm::vec3 const chunkCoordFloored = glm::floor(globalChunkSpacePosition);
        glm::ivec3 const chunkCoord{chunkCoordFloored};

        uint32_t chunkCoordEncoded = util::packIvec8(chunkCoord);
        if(chunkCoordEncoded == 0xffffffff) continue;

        if(chunksInternal.find(chunkCoordEncoded) == chunksInternal.end()) {
            chunksInternal[chunkCoordEncoded].globalPosEncoded = chunkCoordEncoded;
            chunksInternal[chunkCoordEncoded].globalPos = globalChunkSpacePosition;
        }
        
        RawStar raw_star_local = raw_star;

        //BUG
        //TODO: verify fract logic (at some point we got -1.0 to 1.0 here?)
        raw_star_local.mPosition = globalChunkSpacePosition - chunkCoordFloored;

        chunksInternal[chunkCoordEncoded].stars.emplace_back(raw_star_local.mPosition, raw_star_local.mMagnitude, raw_star_local.mTEff);
    }

    // sort rawStars by morton code, this is critical for delta encoding
    for(auto& [id, chunk] : chunksInternal) {
        std::sort(chunk.stars.begin(), chunk.stars.end(),
                  [](StarInternal const& a, StarInternal const& b) { return a.pos < b.pos; });
    }

    // merge overlapping rawStars and write to output
    // walk input from begin to end, appending to output only when new position reached, merging otherwise
    /*for(auto& [id, chunk] : chunksInternal) {
        chunksOut[id] = {};
        chunksOut[id].globalPosEncoded = chunk.globalPosEncoded;
        chunksOut[id].globalPos = chunk.globalPos;
        chunksOut[id].original_count = chunk.rawStars.size();

        chunksOut[id].rawStars.push_back(chunk.rawStars.front());

        auto itIn = chunk.rawStars.begin();
        itIn++; //skip first so we dont add it with itself to output
        for(; itIn != chunk.rawStars.end(); itIn++) {
            Star& lastOutStar = chunksOut[id].rawStars.back();

            if(itIn->pos == lastOutStar.pos) {
                //merge operation
                lastOutStar += *itIn;
                chunksOut[id].merged_count++;
            }
            else {
                //append new position star operation
                chunksOut[id].rawStars.push_back(*itIn);
            }
        }
    }*/

    chunksOut = chunksInternal;
}

//BUG: a problem here (before moving back to non-delta encoded positions) was perhaps that we cacluate row count before accounting for rawStars dropped later?
//fixed in this impl, but remember to re-iterate on the old logic if moving back to deltas

void gpuPrepareChunk(Chunk const& inChunk,
    std::vector<GpuBatchRow>& batchRowsOut,
    std::vector<GpuChunkMeta>& chunkMetaOut
) {
    GpuChunkMeta& meta = chunkMetaOut.emplace_back();
    meta.chunkID     = inChunk.globalPosEncoded;
    meta.startRowNum = batchRowsOut.size();

    constexpr uint32_t PADDING_STAR = 0xFFFFFFFFu;
    std::vector<StarInternal> const& rawStars = inChunk.stars;

    size_t writeIdx = 0; // running output star index, decoupled from input idx

    auto encodeMag = [&](float mag) -> uint16_t {
        //assert(-10.f < mag && mag < 20.f);

        float encodedFloat = (mag + 10.f) / 30.f * static_cast<float>(0xffff);
        return static_cast<uint16_t>(encodedFloat);
    };

    auto encodeTemp = [&](float temp) -> uint16_t {
        //assert (0.f <= temp && temp < 60'000.f);

        return static_cast<uint16_t>(temp);
    };

    auto emitStar = [&](uint32_t pos, float mag, float temp) {
        size_t thread = writeIdx % GPU_THREAD_COUNT;
        if (thread == 0) batchRowsOut.emplace_back();   // start new row on boundary
        GpuBatchRow& row = batchRowsOut.back();

        // TODO: proper uint16_t <-> float mapping for mag/temp
        uint16_t magEnc = encodeMag(mag);
        uint16_t tempEnc = encodeTemp(temp);
        uint32_t magTemp = (uint32_t(magEnc) << 16) | uint32_t(tempEnc);

        row[thread * 2 + 0] = pos;
        row[thread * 2 + 1] = magTemp;
        ++writeIdx;
    };

    // Merge consecutive same-pos rawStars (input is morton-sorted).
    if (!rawStars.empty()) {
        StarInternal acc = rawStars[0];
        for (size_t i = 1; i < rawStars.size(); ++i) {
            if (rawStars[i].pos == acc.pos) {
                float newMag = acc.mag + rawStars[i].mag;
                acc.temp = (newMag > 0.0f)
                    ? (acc.mag * acc.temp + rawStars[i].mag * rawStars[i].temp) / newMag
                    : 0.0f;
                acc.mag  = newMag; // update AFTER computing new temp
            } else {
                emitStar(acc.pos, acc.mag, acc.temp);
                acc = rawStars[i];
            }
        }
        emitStar(acc.pos, acc.mag, acc.temp); // flush last accumulator
    }

    // Pad tail of last row up to the 256-thread boundary.
    while (writeIdx % GPU_THREAD_COUNT != 0) {
        size_t thread = writeIdx % GPU_THREAD_COUNT;
        GpuBatchRow& row = batchRowsOut.back();
        row[thread * 2 + 0] = PADDING_STAR;
        row[thread * 2 + 1] = 0;
        ++writeIdx;
    }

    meta.rowCount = batchRowsOut.size() - meta.startRowNum;
}

//#define REBUILD_CACHE

StarRenderer::StarRenderer(glm::uvec2 const& screenSize)
: mScreenSize{screenSize},
 mAccumulationTexture{createTextureR32F(mScreenSize * glm::uvec2(2,1))}, //twice as wide, interleaved 2-value storage
 mFrameTexture{createTextureRGBA8(mScreenSize)},
mTileCount{glm::uvec2(glm::ceil(glm::vec2(mScreenSize) / glm::vec2(GPU_TILE_SIZE_PX)))},
mTileCount_flat{mTileCount.x * mTileCount.y}
{

    mClearProgram = createComputeProgramFromFile("clear.comp", {});
    mDrawListClearProgram = createComputeProgramFromFile("drawlist_clear.comp", {
        {"MAX_TILE_CHUNKS", GPU_MAX_TILE_CHUNKS}
    });
    mDrawListProgram = createComputeProgramFromFile("drawlist.comp", {
        {"MAX_TILE_CHUNKS", GPU_MAX_TILE_CHUNKS},
        {"TILE_SIZE", GPU_TILE_SIZE_PX}
    });
    mRasterProgram = createComputeProgramFromFile("rasterize.comp", {
        {"MAX_TILE_CHUNKS", GPU_MAX_TILE_CHUNKS},
        {"THREAD_COUNT", GPU_THREAD_COUNT},
        {"TILE_SIZE", GPU_TILE_SIZE_PX}
    });
    mCompositeProgram = createComputeProgramFromFile("composite.comp", {});
    mDebugOverlayProgram = createComputeProgramFromFile("debugoverlay.comp", {});

    mScreenQuadProgram = createProgramFromFiles("quad.vert", "quad.frag");
}

void StarRenderer::preprocessStars(std::vector<RawStar> const& rawStars) {
    #ifdef REBUILD_CACHE
    std::unordered_map<uint32_t, Chunk> chunks;
    bin_stars_to_chunks(rawStars, chunks);

    size_t orig_total = 0;
    size_t actual_total = 0;
    for(auto const& [id, chunk] : chunks) {
        orig_total += chunk.original_count;
        actual_total += chunk.stars.size();
    }

    std::cout << "original total: " << orig_total << "\t actual total: " << actual_total << std::endl;

    std::cout << "total: " << chunks.size() << " chunks" << std::endl;;

    //std::vector<uint32_t> gpuStars;
    std::vector<GpuChunkMeta> gpuChunkMetas;
    gpuChunkMetas.reserve(chunks.size());

    std::vector<GpuBatchRow> gpuBatchRows;
    //copy delta rawStars from chunks sequentially, pass global offset ptr to chunk
    for(auto& [id, chunk] : chunks) {
        gpuPrepareChunk(chunk, gpuBatchRows, gpuChunkMetas);
    }

    std::vector<uint32_t> gpuBatchRowsFlat;
    for(auto const& gpuBatchRow : gpuBatchRows) {
        //serialize main batch table
        for(size_t i = 0; i < GPU_THREAD_COUNT; i++) {
            gpuBatchRowsFlat.push_back(gpuBatchRow[i * 2 + 0]);
            gpuBatchRowsFlat.push_back(gpuBatchRow[i * 2 + 1]);
        }
    }

    //upload data to GPU
    mGpuChunkMetaSSBO.create(gpuChunkMetas, GL_STATIC_DRAW, "data/chunks_galactic.bin");
    mGpuBatchRowsFlatSSBO.create(gpuBatchRowsFlat, GL_STATIC_DRAW, "data/rowsFlat_galactic.bin");
#else
    mGpuChunkMetaSSBO.loadFromFile<GpuChunkMeta>("data/chunks_galactic.bin", GL_STATIC_DRAW);
    mGpuBatchRowsFlatSSBO.loadFromFile<uint32_t>("data/rowsFlat_galactic.bin", GL_STATIC_DRAW);
#endif
    uChunkCount = mGpuChunkMetaSSBO.count();
}

void StarRenderer::prepareGpuBuffers() {
    glGenVertexArrays(1, &mScreenQuadVAOHandle);

    struct TileListTipPointers {
        uint32_t tip_pointers[510];
        uint32_t job_count;
        uint32_t pad0;
    };

    glGenBuffers(1, &mGpuDispatchHeaderSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mGpuDispatchHeaderSSBO);
    // 16 B header + tip pointers
    size_t headerBytes = 16 + mTileCount_flat * sizeof(uint32_t);
    glBufferData(GL_SHADER_STORAGE_BUFFER, headerBytes, nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &mGpuJobsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mGpuJobsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
        16 *    //16 bytes per entry
        (GPU_MAX_TILE_CHUNKS + 1) *  //list entries maximum per tile
        mTileCount_flat * //how many tiles (maybe align to next-highest power of 2?)
        400 // 100x generous allocation for dynamic job list length
        ,
        nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &mGpuProfilingSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mGpuProfilingSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GpuProfilingStruct),
                nullptr, GL_DYNAMIC_READ);
}

namespace {

// Simple: write up to 32 lines of up to 32 chars (newline -> 10), unused cells = 0.
// s: input string, buf: 32*8 uint32_t buffer, returns used lines
size_t fillTextBufferPacked(const std::string &s, uint32_t buf[32*8]) {
    std::fill(buf, buf + 32*8, 0u);
    size_t line = 0, col = 0;              // col = 0..31
    for (unsigned char ch : s) {
        if (line >= 32) break;
        size_t word = line * 8 + (col >> 2);        // which uint32 holds this char
        unsigned shift = (col & 3) * 8;            // byte offset in word
        buf[word] |= (uint32_t)ch << shift;
        if (ch == '\n') { ++line; col = 0; continue; }
        if (++col >= 32) { ++line; col = 0; }
    }
    // count used lines
    size_t used = 0;
    for (; used < 32; ++used) {
        bool any = false;
        for (size_t w = 0; w < 8; ++w) if (buf[used*8 + w]) { any = true; break; }
        if (!any) break;
    }
    return used;
}


}

void StarRenderer::run(glm::vec3 cameraPosition, glm::mat4 modelViewMatrix, glm::mat4 projectionMatrix) {
    // calculate camera matrices
    glm::mat4 uMatMV = (modelViewMatrix);
    glm::mat4 uMatP = (projectionMatrix);

    glm::mat4 uInvP = glm::inverse(uMatP);
    glm::mat4 uViewProj = uMatP * uMatMV;

    //if(mWindow.wasKeyJustPressed(GLFW_KEY_PERIOD)) mExposure *= 5.0;
    //if(mWindow.wasKeyJustPressed(GLFW_KEY_COMMA)) mExposure /= 5.0;
    //TODO: use glProgramUniform instead, so th

    glBindImageTexture(0, mAccumulationTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, mFrameTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);

    static StopWatchGPU sw{};

    //TODO: use glProgramUniform instead, so that we dont have to pass the matrices trough class members
    
    //// PASS: clear screen ////
    sw.startTiming("clear");
    glUseProgram(mClearProgram);
    glUniform2i(glGetUniformLocation(mClearProgram, "uResolution"), mScreenSize.x, mScreenSize.y);
    glDispatchCompute((mScreenSize.x + 7) / 8, (mScreenSize.y + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sw.endTiming("clear");

    //// SSBO BINDING ////
    sw.startTiming("bindings");
    mGpuBatchRowsFlatSSBO.bind(0);
    mGpuChunkMetaSSBO.bind(1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mGpuJobsSSBO); //generated on-gpu (gpu-driven rendering)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, mGpuDispatchHeaderSSBO);
    sw.endTiming("bindings");

    //// PASS: clear draw list ////
    sw.startTiming("drawlist_clear");
    glUseProgram(mDrawListClearProgram);
    glUniform1ui(glGetUniformLocation(mDrawListClearProgram, "uTotalTileCount"), mTileCount_flat);
    glDispatchCompute((mTileCount_flat + 31) / 32, 1, 1); //for each screentile //TODO: remove 2x overshoot
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sw.endTiming("drawlist_clear");

    //// PASS: generate draw list ////
    sw.startTiming("drawlist");
    glUseProgram(mDrawListProgram);
    glUniform1ui(glGetUniformLocation(mDrawListProgram, "uChunkCount"), uChunkCount);
    glUniform1f(glGetUniformLocation(mDrawListProgram, "uChunkSize"), CHUNK_SIZE_PARSECS);
    glUniform2i(glGetUniformLocation(mDrawListProgram, "uResolution"), mScreenSize.x, mScreenSize.y);
    
    static double max = 8;
    //if(mWindow.wasKeyJustPressed(GLFW_KEY_1)) max /= 1.5;
    //if(mWindow.wasKeyJustPressed(GLFW_KEY_2)) max *= 1.5;
    //if(max < 1) max = 1;

    uint32_t m = static_cast<uint32_t>(max);

    glUniform1ui(glGetUniformLocation(mDrawListProgram, "uMAX_JOB_WORK"), m * 2);
    glUniform1ui(glGetUniformLocation(mDrawListProgram, "uMIN_CHUNK_WORK"), m / 2);
    glUniform1ui(glGetUniformLocation(mDrawListProgram, "uMAX_CHUNK_WORK"), m);

    glUniformMatrix4fv(glGetUniformLocation(mDrawListProgram, "uViewProj"), 1, GL_FALSE, &uViewProj[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(mDrawListProgram, "uInvP"), 1, GL_FALSE, &uInvP[0][0]);
    glDispatchCompute((uChunkCount + 63) / 64, 1, 1); //for each screentile
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
    sw.endTiming("drawlist");


    { // prepare profiling ssbo
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mGpuProfilingSSBO);
        const uint32_t zero = 0;
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI,
                        GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mGpuProfilingSSBO);
    }

    //// PASS: main rasterization ////
    glm::vec3 camPos = cameraPosition;

    sw.startTiming("raster");
    glUseProgram(mRasterProgram);
    glUniform3f(glGetUniformLocation(mRasterProgram, "uObserverPosWorld"), camPos.x, camPos.y, camPos.z);
    glUniform2i(glGetUniformLocation(mRasterProgram, "uResolution"), mScreenSize.x, mScreenSize.y);
    glUniform1f(glGetUniformLocation(mRasterProgram, "uChunkSize"), CHUNK_SIZE_PARSECS);
    glUniformMatrix4fv(glGetUniformLocation(mRasterProgram, "uViewProj"), 1, GL_FALSE, &uViewProj[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(mRasterProgram, "uInvP"), 1, GL_FALSE, &uInvP[0][0]);
    glUniform1ui(glGetUniformLocation(mRasterProgram, "uStartLayer"), 0);
    glUniform1ui(glGetUniformLocation(mRasterProgram, "uLayerCount"), GPU_MAX_TILE_CHUNKS);

    //indirect dispatch
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, mGpuDispatchHeaderSSBO);
    glDispatchComputeIndirect(0);   // x/y/z read from offset 0 of the bound buffer
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    sw.endTiming("raster");

    uint32_t gpuJobDispatchCount = 0;

    { // read back profiling ssbo from GPU
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                   sizeof(GpuProfilingStruct), &mProfilingCPU);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
            sizeof(uint32_t), &gpuJobDispatchCount);
    }

    //// PASS: post-render composition ////
    sw.startTiming("compose");
    glUseProgram(mCompositeProgram);
    glUniform2i(glGetUniformLocation(mCompositeProgram, "uResolution"), mScreenSize.x, mScreenSize.y);
    glUniform1f(glGetUniformLocation(mCompositeProgram, "uExposure"), mExposure);
    glDispatchCompute((mScreenSize.x + 7) / 8, (mScreenSize.y + 7) / 8, 1); //tiling like clear program
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    sw.endTiming("compose");

    //// PASS: debug text overlay ////
    //if(!mDebugOverlayEnabled) return;
    std::stringstream ss;
    ss << "clear:    " << std::setw(6) << sw.getMeasuredMicroseconds("clear") << '\n';
    ss << "drawlist_clear: " << std::setw(6) << sw.getMeasuredMicroseconds("drawlist_clear") << '\n';
    ss << "bindings: " << std::setw(6) << sw.getMeasuredMicroseconds("bindings") << '\n';
    ss << "drawlist: " << std::setw(6) << sw.getMeasuredMicroseconds("drawlist") << '\n';
    ss << "raster:   " << std::setw(6) << sw.getMeasuredMicroseconds("raster") << '\n';
    ss << "compose:  " << std::setw(6) << sw.getMeasuredMicroseconds("compose") << '\n';
    ss << '\n';
    ss << "-----------" << '\n';
    ss << "rawStars:    " << std::setw(9) << mProfilingCPU.starsDrawn  << '\n';
    ss << "s.total:  " << std::setw(9) << mProfilingCPU.starsTotal  << '\n';
    ss << "chunks:   " << std::setw(5) << mProfilingCPU.chunksDrawn << '\n';
    ss << "tiles:    " << std::setw(5) << mProfilingCPU.tilesDrawn  << '\n';
    double microSecondsPer1M = static_cast<double>(sw.getMeasuredMicroseconds("raster")) / (static_cast<double>(mProfilingCPU.starsTotal) / 1'000'000.0);
    ss << "per. 1m rawStars: " << std::setw(6) << microSecondsPer1M << '\n';
    ss << "job count: " << std::setw(6) << mProfilingCPU.jobCount << '\n';
    ss << '\n';
    ss << "-----------" << '\n';
    ss << "camera x: " << std::setw(10) << cameraPosition.x << '\n';
    ss << "camera y: " << std::setw(10) << cameraPosition.y << '\n';
    ss << "camera z: " << std::setw(10) << cameraPosition.z << '\n';
    ss << "-----------" << '\n';
    //ss << "max:" << max << '\n';

    std::cout << glm::to_string(uMatMV) << std::endl;
    std::cout << glm::to_string(uMatP) << std::endl;


    uint32_t charBuf[32*8] = {0};
    std::string ss_str = ss.str();
    fillTextBufferPacked(ss.str(), charBuf);

    glUseProgram(mDebugOverlayProgram);
    glUniform2i(glGetUniformLocation(mDebugOverlayProgram, "uResolution"), mScreenSize.x, mScreenSize.y);
    glUniform1ui(glGetUniformLocation(mDebugOverlayProgram, "uLineCount"), 32);
    glUniform1uiv(glGetUniformLocation(mDebugOverlayProgram, "uLineChars"), 32 * 8, static_cast<GLuint const * const>(charBuf));
    glDispatchCompute(32, 1, 1); //one work group per line, one thread per char
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    std::cout << mScreenQuadProgram << std::endl;
    std::cout << mScreenQuadVAOHandle << std::endl;
    std::cout << mFrameTexture << std::endl;

    glUseProgram(mScreenQuadProgram);
    glBindVertexArray(mScreenQuadVAOHandle);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mFrameTexture);
    glUniform1i(glGetUniformLocation(mScreenQuadProgram, "uColorTex"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    GLenum e = glGetError();
    std::cout << "GL err 0x" << std::hex << e << '\n';
}


StarRenderer::~StarRenderer() {
    glDeleteBuffers(1, &mGpuJobsSSBO);
    glDeleteTextures(1, &mAccumulationTexture);
    glDeleteTextures(1, &mFrameTexture);

    glDeleteVertexArrays(1, &mScreenQuadVAOHandle);

    glDeleteProgram(mClearProgram);
    glDeleteProgram(mDrawListClearProgram);
    glDeleteProgram(mDrawListProgram);
    glDeleteProgram(mRasterProgram);
    glDeleteProgram(mCompositeProgram);
    glDeleteProgram(mDebugOverlayProgram);
    glDeleteProgram(mScreenQuadProgram);
}

