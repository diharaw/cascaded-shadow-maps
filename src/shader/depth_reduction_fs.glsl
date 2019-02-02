// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec2 PS_OUT_Color;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec2 PS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2D s_Texture;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec4 depth_x = textureGather(s_Texture, PS_IN_TexCoord, 0);
    vec4 depth_y = textureGather(s_Texture, PS_IN_TexCoord, 1);

	PS_OUT_Color = vec2(min(min(depth_x.x, depth_x.y), min(depth_x.z, depth_x.w)), max(max(depth_y.x, depth_y.y), max(depth_y.z, depth_y.w)));
}

// ------------------------------------------------------------------