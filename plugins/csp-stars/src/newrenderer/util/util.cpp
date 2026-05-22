#include "util.hpp"

#include <random>
#include <cstddef>
#include <chrono>
#include <fstream>
#include <iostream>

namespace util {
uint32_t rand_u32() {
    thread_local static std::mt19937 engine{std::random_device{}()};
    thread_local static std::uniform_int_distribution<uint32_t> dist;
    return dist(engine);
}

uint32_t hash_u32(uint32_t v) {
    std::size_t h = std::hash<uint32_t>{}(v);
    return static_cast<uint32_t>(h);
}


double ema(double value, double alpha) {
    static double avg = 0.0;
    static int countBeforeStarting = 10;
    if(countBeforeStarting == 2) avg = value;
    if (countBeforeStarting > 0) {
        countBeforeStarting--;
        return 0.0;
    }

    avg = (1.0 - alpha) * avg + alpha * value;
    return avg;
}

double time_now() {
    using clock = std::chrono::steady_clock;
    const auto tp = clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::duration<double>>(tp.time_since_epoch());
    return ms.count();
}

//from https://github.com/tomas-o-dev/Morton-Z-Code-C-library/blob/master/MZC3D32.h [MIT License]

uint32_t ulMC3Dspread(uint32_t w) {
    //w &=                0x000003ff; /* w = ---- ---- ---- ---- ---- --98 7654 3210 */
    w &= 0xff;
    w = (w | w << 16) & 0x030000ff; 
    w = (w | w <<  8) & 0x0300f00f;
    w = (w | w <<  4) & 0x030c30c3;
    w = (w | w <<  2) & 0x09249249; /* w = uu-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0 */
    return w;
    }

/* inverse of Spread */
uint32_t ulMC3Dcompact(uint32_t w)  {
	 w &=                  0x09249249;
	 w = (w ^ (w >>  2)) & 0x030c30c3;
	 w = (w ^ (w >>  4)) & 0x0f00f00f;
	 w = (w ^ (w >>  8)) & 0xff0000ff;
	 w = (w ^ (w >> 16)) & 0x0000ffff;
     w &= 0xFF;
	 return (uint32_t)w;
}

uint32_t encode_morton(glm::uvec3 pos) {
    uint32_t x = pos.x & 0xFF;
    uint32_t y = pos.y & 0xFF;
    uint32_t z = pos.z & 0xFF;
    
    x = ulMC3Dspread(x);
    y = ulMC3Dspread(y) << 1;
    z = ulMC3Dspread(z) << 2;
    
    return x | y | z;
}

glm::uvec3 decode_morton(uint32_t morton) {
    uint32_t x,y,z{};

    x = ulMC3Dcompact(morton);
    y = ulMC3Dcompact(morton >> 1);    
    z = ulMC3Dcompact(morton >> 2);    
    
    return glm::uvec3{x,y,z};
}

bool saveVec4Array(const std::string& filepath, const std::vector<glm::vec4>& vecs) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    uint32_t count = static_cast<uint32_t>(vecs.size());
    
    // Write size header
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // Write all vectors
    if (count > 0) {
        file.write(reinterpret_cast<const char*>(vecs.data()), 
                   count * sizeof(glm::vec4));
    }
    
    return file.good();
}

bool loadVec4Array(const std::string& filepath, std::vector<glm::vec4>& vecs) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    uint32_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file) return false;
    
    vecs.resize(count);
    if (count > 0) {
        file.read(reinterpret_cast<char*>(vecs.data()), 
                  count * sizeof(glm::vec4));
    }
    
    return file.good();
}

uint32_t packIvec8(glm::ivec3 pos) {
    constexpr int32_t offset = 0x7f;
    if(std::abs(pos.x) > offset || std::abs(pos.y) > offset || std::abs(pos.z) > offset) {
        return 0xffffffff;
        //std::cout << "assertion failed in packIvec8" << std::endl;
        //exit(-1);
    }

    //glm::uvec3 shifted = pos + glm::ivec3(offset);

    uint32_t x = (pos.x + offset) & 0xff;
    uint32_t y = (pos.y + offset) & 0xff;
    uint32_t z = (pos.z + offset) & 0xff;

    return (x << 16) | (y << 8) | z;
}

glm::ivec3 unpackIvec8(uint32_t v) {
    constexpr int32_t offset = 0x7f;

    uint8_t x = (v >> 16) & 0xff;
    uint8_t y = (v >> 8) & 0xff;
    uint8_t z = (v >> 0) & 0xff;

    glm::ivec3 pos{x,y,z};
    glm::ivec3 unshifted = pos - glm::ivec3(offset);

    return unshifted;
}

} //namespace util