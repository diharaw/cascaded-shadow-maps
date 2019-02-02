// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec3 PS_IN_WorldFragPos;
out vec4 PS_IN_NDCFragPos;
out vec3 PS_IN_Normal;
out vec2 PS_IN_TexCoord;

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
    vec4 position = model * vec4(VS_IN_Position, 1.0);
	PS_IN_WorldFragPos = position.xyz;
	PS_IN_Normal = mat3(model) * VS_IN_Normal;
	PS_IN_TexCoord = VS_IN_TexCoord;
    PS_IN_NDCFragPos = projection * view * position;
    gl_Position = PS_IN_NDCFragPos;
}

// ------------------------------------------------------------------