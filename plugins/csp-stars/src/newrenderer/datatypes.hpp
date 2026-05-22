#pragma once

#include <glm/glm.hpp>

struct TemperatureColor {
    uint8_t l_warm; //maybe encode in exponential for better HDR depth?
    uint8_t l_cold; //maybe encode in exponential for better HDR depth?
};

struct StarInternal { //32 bit lowres representation
private:
    uint32_t encodeLocalPosition(glm::vec3 const& localPosition) {
        // clamp to [0,1] to avoid overflow
        glm::vec3 p = glm::clamp(localPosition, 0.0f, 1.0f);

        // convert to 10-bit integers
        uint32_t x = uint32_t(p.x * 1023.0f + 0.0f);
        uint32_t y = uint32_t(p.y * 1023.0f + 0.0f);
        uint32_t z = uint32_t(p.z * 1023.0f + 0.0f);

        return (x << 20) | (y << 10) | (z << 0);

        //return util::encode_morton(localPosition * 64.f);
    }
public:
    // encode position as 32 bit morton code, not delta yet
    uint32_t pos = 0;
    float mag = 0.0;
    float temp = 0.0;

    StarInternal(glm::vec3 const& localPosition, float magnitude, float temperature
    ) :
    pos {encodeLocalPosition(localPosition)},
    mag{magnitude},
    temp{temperature}
    {}

    StarInternal& operator+=(StarInternal const& other) {
        float newMag = mag + other.mag;
        if (newMag > 0.0f)
            temp = (mag * temp + other.mag * other.temp) / newMag;
        mag = newMag;
        return *this;
    }
};

struct Chunk {
    uint32_t globalPosEncoded = 0; //packed world position
    glm::ivec3 globalPos{}; //non-packed world position, for easier access
    std::vector<StarInternal> stars{};

    //auxilliary statistical counters
    uint32_t merged_count = 0;
    uint32_t original_count = 0;
};