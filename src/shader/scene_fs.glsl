// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

out vec4 PS_OUT_Color;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

in vec3 PS_IN_WorldFragPos;
in vec4 PS_IN_NDCFragPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std140, binding = 2) buffer CSMUniforms
{
    vec4 direction;
    vec4 options;
    int num_cascades;
    float far_bounds[8];
    mat4 texture_matrices[8];
};

uniform sampler2D s_Diffuse;
uniform sampler2DArray s_ShadowMap;

// ------------------------------------------------------------------

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

// ------------------------------------------------------------------

float shadow_occlussion(float frag_depth, vec3 n, vec3 l)
{
	int index = 0;
    float blend = 0.0;
    
	// Find shadow cascade.
	for (int i = 0; i < num_cascades - 1; i++)
	{
		if (frag_depth > far_bounds[i])
			index = i + 1;
	}

	blend = clamp( (frag_depth - far_bounds[index] * 0.995) * 200.0, 0.0, 1.0);
    
    // Apply blend options.
    blend *= options.z;

	// Transform frag position into Light-space.
	vec4 light_space_pos = texture_matrices[index] * vec4(PS_IN_WorldFragPos, 1.0f);

	float current_depth = light_space_pos.z;
    
	float bias = max(0.0005 * (1.0 - dot(n, l)), 0.0005);  

	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(s_ShadowMap, 0).xy;
	for(int x = -1; x <= 1; ++x)
	{
	    for(int y = -1; y <= 1; ++y)
	    {
	        float pcfDepth = texture(s_ShadowMap, vec3(light_space_pos.xy + vec2(x, y) * texelSize, float(index))).r; 
	        shadow += current_depth - bias > pcfDepth ? 1.0 : 0.0;        
	    }    
	}
	shadow /= 9.0;
	
    if (options.x == 1.0)
    {
        //if (blend > 0.0 && index != num_cascades - 1)
        //{
        //    light_space_pos = texture_matrices[index + 1] * vec4(PS_IN_WorldFragPos, 1.0f);
        //    shadow_map_depth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index + 1))).r;
        //    current_depth = light_space_pos.z;
        //    float next_shadow = depth_compare(current_depth, shadow_map_depth, bias);
        //    
        //    return (1.0 - blend) * shadow + blend * next_shadow;
        //}
        //else
		    return shadow;
    }
    else
        return 0.0;
}

// ------------------------------------------------------------------

vec3 debug_color(float frag_depth)
{
	int index = 0;

	// Find shadow cascade.
	for (int i = 0; i < num_cascades - 1; i++)
	{
		if (frag_depth > far_bounds[i])
			index = i + 1;
	}

	if (index == 0)
		return vec3(1.0, 0.0, 0.0);
	else if (index == 1)
		return vec3(0.0, 1.0, 0.0);
	else if (index == 2)
		return vec3(0.0, 0.0, 1.0);
	else
		return vec3(1.0, 1.0, 0.0);
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	vec3 n = normalize(PS_IN_Normal);
	vec3 l = -direction.xyz;

	float lambert = max(0.0f, dot(n, l));

	vec3 diffuse = vec3(0.7);// texture(s_Diffuse, PS_IN_TexCoord * 50).xyz;
	vec3 ambient = diffuse * 0.3;

	float frag_depth = (PS_IN_NDCFragPos.z / PS_IN_NDCFragPos.w) * 0.5 + 0.5;
	float shadow = shadow_occlussion(frag_depth, n, l);
	
    vec3 cascade = options.y == 1.0 ? debug_color(frag_depth) : vec3(0.0);
	vec3 color = (1.0 - shadow) * diffuse * lambert + ambient + cascade * 0.5;

    PS_OUT_Color = vec4(color, 1.0);
}

// ------------------------------------------------------------------