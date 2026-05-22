#pragma once

#include "ssbo.hpp"
#include <vector>
#include <string>

#include <glm/glm.hpp>

class StarRenderer {
public:

    // RawStar structure matching Cosmoscout Stars::mStars structure
    struct RawStar {
        float mMagnitude;               // Apparent magnitude
        float mTEff;                    // Effective temperature (K)
        glm::highp_f32vec3 mPosition;     // XYZ position in parsecs, with f32 precision //TODO: switch to high precision coordinates
    };

    //TODO add chunk size as test parameter (limit not on distance in chunks, but in parsecs)
    StarRenderer(glm::uvec2 const& screenSize);
    void preprocessStars(std::vector<RawStar> const& rawStars);
    void prepareGpuBuffers();

    ~StarRenderer();

    void run(glm::vec3 cameraPos, glm::mat4 modelViewMatrix, glm::mat4 projectionMatrix);
    GLuint getFrameTexture();
private:
    glm::uvec2 const mScreenSize; //TODO: handle viewport resizing

private:
    GLuint mClearProgram;
    GLuint mDrawListClearProgram;
    GLuint mDrawListProgram;
    GLuint mRasterProgram;
    GLuint mCompositeProgram;
    GLuint mDebugOverlayProgram;

    GLuint mScreenQuadProgram;

    GLuint mScreenQuadVAOHandle;

private: //framebuffers
    GLuint mAccumulationTexture; //for accumulating interleaved 32bit luminance-weighted temperature + 32 bit luminance
    GLuint mFrameTexture;
private:
    GLuint mGpuJobsSSBO;
    GLuint mGpuDispatchHeaderSSBO;
    SSBO mGpuChunkMetaSSBO;
    SSBO mGpuBatchRowsFlatSSBO;

private: //profiling SSBO
    struct GpuProfilingStruct {
        uint32_t starsDrawn;
        uint32_t chunksDrawn;
        uint32_t tilesDrawn;
        uint32_t starsTotal;
        uint32_t jobCount;
    };
    GLuint mGpuProfilingSSBO;
    GpuProfilingStruct mProfilingCPU{};

private:
    size_t uChunkCount;

private:
    glm::uvec2 const mTileCount;
    size_t const mTileCount_flat;

private:
    bool mDebugOverlayEnabled = false;

    float mExposure =  1e6;
};
