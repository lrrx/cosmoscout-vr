#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <string>

namespace util {
uint32_t rand_u32();
uint32_t hash_u32(uint32_t v);

double ema(double value, double alpha = 0.1);
double time_now();

uint32_t encode_morton(glm::uvec3 pos);
glm::uvec3 decode_morton(uint32_t morton);

uint32_t packIvec8(glm::ivec3 pos);
glm::ivec3 unpackIvec8(uint32_t v);

bool saveVec4Array(const std::string& filepath, const std::vector<glm::vec4>& vecs);
bool loadVec4Array(const std::string& filepath, std::vector<glm::vec4>& vecs);

}