#pragma once

#include <string>
#include <array>
#include <unordered_map>

#include <GL/glew.h>

class StopWatchGPU {
public:
    StopWatchGPU() = default;
    ~StopWatchGPU() {
        for (auto& [tag, slot] : slots) {
            glDeleteQueries(static_cast<GLsizei>(slot.queries.size()),
                            slot.queries.data());
        }
    }

    StopWatchGPU(StopWatchGPU const&) = delete;
    StopWatchGPU& operator=(StopWatchGPU const&) = delete;

    void startTiming(std::string const& tag) {
        Slot& s = getOrCreate(tag);
        glQueryCounter(s.queries[s.writeIdx * 2 + 0], GL_TIMESTAMP);
    }

    void endTiming(std::string const& tag) {
        auto it = slots.find(tag);
        if (it == slots.end()) return; // end without start
        Slot& s = it->second;

        glQueryCounter(s.queries[s.writeIdx * 2 + 1], GL_TIMESTAMP);

        // advance write head; after wrap, the new writeIdx points at the oldest slot
        s.writeIdx = (s.writeIdx + 1) % RING_SIZE;
        if (s.framesInFlight < RING_SIZE) ++s.framesInFlight;

        // once the ring is full, harvest the oldest pair (>=2 frames old, should be ready)
        if (s.framesInFlight < RING_SIZE) return;

        size_t readIdx = s.writeIdx;
        GLuint qs = s.queries[readIdx * 2 + 0];
        GLuint qe = s.queries[readIdx * 2 + 1];

        GLint available = 0;
        glGetQueryObjectiv(qe, GL_QUERY_RESULT_AVAILABLE, &available);
        if (!available) return; // skip sample rather than stall

        GLuint64 t0 = 0, t1 = 0;
        glGetQueryObjectui64v(qs, GL_QUERY_RESULT, &t0);
        glGetQueryObjectui64v(qe, GL_QUERY_RESULT, &t1);
        double seconds = static_cast<double>(t1 - t0) * 1e-9;

        constexpr double alpha = 0.1;
        if (!s.hasValue) {
            s.smoothed = seconds;
            s.hasValue = true;
        } else {
            s.smoothed = (1.0 - alpha) * s.smoothed + alpha * seconds;
        }
    }

    uint32_t getMeasuredMicroseconds(std::string const& tag) {
        auto it = slots.find(tag);
        if (it == slots.end() || !it->second.hasValue) return 0;
        return static_cast<uint32_t>(it->second.smoothed * 1'000'000.0);
    }

private:
    static constexpr size_t RING_SIZE = 5;

    struct Slot {
        std::array<GLuint, RING_SIZE * 2> queries{}; // [s0,e0, s1,e1, s2,e2]
        size_t writeIdx = 0;
        size_t framesInFlight = 0;
        double smoothed = 0.0;
        bool hasValue = false;
    };

    Slot& getOrCreate(std::string const& tag) {
        auto it = slots.find(tag);
        if (it != slots.end()) return it->second;
        Slot& s = slots[tag];
        glGenQueries(static_cast<GLsizei>(s.queries.size()), s.queries.data());
        return s;
    }

    std::unordered_map<std::string, Slot> slots;
};