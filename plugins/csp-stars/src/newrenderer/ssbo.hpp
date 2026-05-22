#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include <GL/glew.h>

// Simple SSBO wrapper with RAII
class SSBO {
public:
    SSBO() : mCount(0), m_handle(0) {}

    ~SSBO() {
        cleanup();
    }

    // Prevent copying
    SSBO(const SSBO&) = delete;
    SSBO& operator=(const SSBO&) = delete;

    // don' allow moving for now
    SSBO(SSBO&& other) = delete;
    SSBO& operator=(SSBO&& other) = delete;
    /*SSBO(SSBO&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = 0;
    }

    SSBO& operator=(SSBO&& other) noexcept {
        if (this != &other) {
            cleanup();
            m_handle = other.m_handle;
            other.m_handle = 0;
        }
        return *this;
    }*/

    // Create and upload data
    template<typename T>
    void create(const std::vector<T>& data, GLenum usage, std::string const& cache_filename = "") {
        static_assert(std::is_trivially_copyable_v<T>, "SSBO data must be trivially copyable");
        
        if(cache_filename.length() > 0) {
            std::ofstream out(cache_filename, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open file for writing: " + cache_filename);
            out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        }

        mCount = data.size();
        cleanup();
        glGenBuffers(1, &m_handle);
        bind();
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.size() * sizeof(T), data.data(), usage);
    }

    size_t count() const {
        return mCount;
    }

    // Bind to a specific shader storage buffer slot
    void bind(GLuint index = 0) const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, m_handle);
    }

    // Unbind
    void unbind() const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    }

    // Check if buffer is valid
    bool isValid() const {
        return m_handle != 0;
    }

    // Get buffer ID
    GLuint getID() const {
        return m_handle;
    }

    template<typename T>
    void loadFromFile(std::string filename, GLenum usage) {
        static_assert(std::is_trivially_copyable_v<T>, "SSBO data must be trivially copyable");

        std::ifstream in(filename, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("Failed to open file for reading: " + filename);

        std::streamsize size = in.tellg();
        if (size % sizeof(T) != 0) throw std::runtime_error("File size does not match type size");
        in.seekg(0, std::ios::beg);

        std::vector<T> data(size / sizeof(T));
        if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
            throw std::runtime_error("Failed to read file data");
        }

        create(data, usage);
    }

private:
    void cleanup() {
        if (m_handle != 0) {
            glDeleteBuffers(1, &m_handle);
            m_handle = 0;
            mCount = 0;
        }
    }

    size_t mCount;
    GLuint m_handle;
};
