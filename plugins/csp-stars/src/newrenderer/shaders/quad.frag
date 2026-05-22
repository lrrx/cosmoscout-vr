#version 460 core

out vec4 FragColor;

uniform sampler2D uColorTex;

void main() {
    ivec2 fragCoord = ivec2(gl_FragCoord.xy);

    vec4 texel = texelFetch(uColorTex, fragCoord, 0);
    if(fragCoord.x == 0 || fragCoord.y == 0) texel = vec4(1.0);
    if(fragCoord.x == 4 || fragCoord.y == 4) texel = vec4(1.0);

    FragColor = texel;
}
