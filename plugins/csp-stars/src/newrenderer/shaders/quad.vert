#version 460 core

out gl_PerVertex {
    vec4 gl_Position;
};


void main() {
    // 6-vertex fullscreen quad (two triangles) via gl_VertexID 0..5
    const vec2 P[6] = vec2[](
    vec2(-1.0, -1.0), // 0
    vec2(-1.0,  1.0), // 1 (swapped)
    vec2( 1.0, -1.0), // 2 (swapped)
    vec2(-1.0,  1.0), // 3
    vec2( 1.0,  1.0), // 4 (swapped)
    vec2( 1.0, -1.0)  // 5 (swapped)
    );

    vec2 pos = P[gl_VertexID % 6]; // derive UV from position: map from NDC (-1..1) to UV (0..1)

    gl_Position = vec4(pos, 0.0, 1.0);
}
