#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "shader_utils.hpp"

std::string loadFile(std::string const& name)
{

    constexpr auto SHADER_BASE_PATH = "../share/newrenderer/shaders/";
    std::string const path = SHADER_BASE_PATH + name;

    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open file: " << path << "\n";
        return {};
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

GLuint compileShader(GLenum type, const std::string& src)
{
    GLuint shader = glCreateShader(type);
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, 2048, nullptr, log);
        std::cout << src << std::endl;
        std::cerr << "Shader compile error:\n" << log << "\n";
        std::quick_exit(-1);
    }
    return shader;
}

GLuint createProgramFromFiles(std::string const& vsname, std::string const& fsname)
{
    std::string vsrc = loadFile(vsname);
    std::string fsrc = loadFile(fsname);

    std::cout << vsrc << std::endl;
    std::cout << fsrc << std::endl;

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, 2048, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
        std::quick_exit(-1);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void injectDefine(std::string& shaderSource, std::string const& name, size_t value) {
        // Inject WARP_SIZE define after the #version directive
    std::string define = "#define " + name + " " + std::to_string(value) + "\n";
    size_t versionPos = shaderSource.find("#version");
    if (versionPos != std::string::npos) {
        size_t firstNewline = shaderSource.find('\n', versionPos);
        if (firstNewline != std::string::npos) {
            shaderSource.insert(firstNewline + 1, define);
        } else {
            shaderSource += "\n" + define;
        }
    } else {
        shaderSource = define + shaderSource;
    }
}

//#define DUMP_COMPUTE_SHADER_BINARY

GLuint createComputeProgramFromFile(std::string const& csname, std::unordered_map<std::string, size_t> const& defines)
{
    std::string csrc = loadFile(csname);

    for(auto const& [name, value] : defines){
        injectDefine(csrc, name, value);
    }

    GLuint cs = compileShader(GL_COMPUTE_SHADER, csrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(prog, 2048, nullptr, log);
        std::cerr << "Compute program link error:\n" << log << "\n";
        std::quick_exit(-1);
    }
#ifdef DUMP_COMPUTE_SHADER_BINARY
    // add this once after link succeeds in createComputeProgramFromFile
    GLint binLen = 0;
    glGetProgramiv(prog, GL_PROGRAM_BINARY_LENGTH, &binLen);
    std::vector<char> bin(binLen);
    GLenum format = 0;
    glGetProgramBinary(prog, binLen, nullptr, &format, bin.data());
    std::ofstream(std::to_string(prog) + "shader_dump.bin", std::ios::binary).write(bin.data(), binLen);
#endif

    glDeleteShader(cs);
    return prog;
}
