#pragma once
#include <string>
#include <unordered_map>

#include <GL/glew.h>

GLuint createProgramFromFiles(std::string const& vsPath, std::string const& fsPath);
GLuint createComputeProgramFromFile(std::string const& csPath, std::unordered_map<std::string, size_t> const& defines);
