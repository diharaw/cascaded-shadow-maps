// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 0) buffer GlobalUniforms 
{
    mat4 view;
    mat4 projection;
    mat4 crop[8];
};

layout (std140) uniform ObjectUniforms
{
    mat4 model;
};

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    gl_Position = projection * view * model * vec4(VS_IN_Position, 1.0);
}

// ------------------------------------------------------------------