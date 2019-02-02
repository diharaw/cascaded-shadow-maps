// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = 1, local_size_y = 1) in;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 0) buffer GlobalUniforms
{
    mat4 view;
    mat4 projection;
    mat4 crop[8];
};

layout(std140, binding = 1) buffer CSMUniforms
{
	mat4 texture_matrices[8];
    vec4 direction;
    vec4 options;
    int num_cascades;
    float far_bounds[8];
};

uniform float u_Near;
uniform float u_Far;
uniform vec3 u_CameraPos;
uniform vec3 u_CameraDir;
uniform vec3 u_CameraUp;
uniform int u_MaxMip;
uniform float u_Lambda;
uniform float u_NearOffset;
uniform float u_FOV;
uniform float u_Ratio;
uniform mat4 u_Bias;
uniform mat4 u_ModelView;
uniform int u_StablePSSM;
uniform int u_ShadowMapSize;

uniform sampler2D u_Depth;

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct FrustumSplit
{
	float near_plane;
	float far_plane;
	float ratio;
	float fov;
	vec3 center;
	vec3 corners[8];
};

// ------------------------------------------------------------------
// GLOBALS ----------------------------------------------------------
// ------------------------------------------------------------------

#define MAX_FRUSTUM_SPLITS 8
#define kINFINITY 999999999

FrustumSplit splits[MAX_FRUSTUM_SPLITS];
mat4 proj_matrices[MAX_FRUSTUM_SPLITS];

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

mat4 look_at(vec3 _eye, vec3 _origin, vec3 _up)
{
	vec3 front = normalize(_origin - _eye);
	vec3 right = normalize(cross(front, _up));
	vec3 up = normalize(cross(right, front));

	mat4 m = mat4(1.0);

	m[0][0] = right.x;
	m[1][0] = right.y;
	m[2][0] = right.z;

	m[0][1] = up.x;
	m[1][1] = up.y;
	m[2][1] = up.z;

	m[0][2] = -front.x;
	m[1][2] = -front.y;
	m[2][2] = -front.z;

	m[3][0] = -dot(right, _eye);
	m[3][1] = -dot(up, _eye);
	m[3][2] = dot(front, _eye);

	return m;
}

// ------------------------------------------------------------------

mat4 ortho(float _l, float _r, float _b, float _t, float _n, float _f)
{
	mat4 m = mat4(1.0);

	m[0][0] = 2.0 / (_r - _l);
	m[1][1] = 2.0 / (_t - _b);
	m[2][2] = -2.0 / (_f - _n);

	m[3][0] = -(_r + _l) / (_r - _l);
	m[3][1] = -(_t + _b) / (_t - _b);
	m[3][2] = -(_f + _n) / (_f - _n);

	return m;
}

// ------------------------------------------------------------------

// Take exponential depth and convert into linear depth.

float depth_exp_to_view(mat4 inverse_proj, float exp_depth)
{
    exp_depth = exp_depth * 2.0 - 1.0;
    float w = inverse_proj[2][3] * exp_depth + inverse_proj[3][3];
    return (1.0 / w);
}

// ------------------------------------------------------------------

float depth_exp_to_view(float near, float far, float exp_depth)
{
   return (2.0 * near * far) / (far + near - exp_depth * (far - near));
}

// ------------------------------------------------------------------

float depth_view_to_linear_01(float near, float far, float depth)
{
	return (depth - near) / (far - near);
}

// ------------------------------------------------------------------

float depth_linear_01_to_view(float near, float far, float depth)
{
	return near + depth * (far - near);
}

// ------------------------------------------------------------------

float depth_exp_to_linear_01(float near, float far, float depth)
{
    float view_depth = depth_exp_to_view(near, far, depth);
    return depth_view_to_linear_01(near, far, view_depth);
}

// ------------------------------------------------------------------

void update_splits()
{
    vec2 min_max = textureLod(u_Depth, vec2(0.0), u_MaxMip).xy;

	// float nd = depth_exp_to_view(u_Near, u_Far, min_max.x) - 0.1;
	float fd = depth_exp_to_view(u_Near, u_Far, min_max.y) + 1.0;

	float nd = u_Near;
	// float fd = 200.0;

	float lambda = u_Lambda;
	float ratio = fd / nd;
	splits[0].near_plane = nd;

	for (int i = 1; i < num_cascades; i++)
	{
		float si = i / float(num_cascades);

		// Practical Split Scheme: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
		float t_near = lambda * (nd * pow(ratio, si)) + (1 - lambda) * (nd + (fd - nd) * si);
		float t_far = t_near * 1.005;
		splits[i].near_plane = t_near;
		splits[i - 1].far_plane = t_far;
	}

	splits[num_cascades - 1].far_plane = fd;
}

// ------------------------------------------------------------------

void update_frustum_corners()
{
	vec3 center = u_CameraPos;
	vec3 view_dir = u_CameraDir;

	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = cross(view_dir, up);

	for (int i = 0; i < num_cascades; i++)
	{
		vec3 fc = center + view_dir * splits[i].far_plane;
		vec3 nc = center + view_dir * splits[i].near_plane;

		right = normalize(right);
		up = normalize(cross(right, view_dir));

		// these heights and widths are half the heights and widths of
		// the near and far plane rectangles
		float near_height = tan(splits[i].fov / 2.0f) * splits[i].near_plane;
		float near_width = near_height * splits[i].ratio;
		float far_height = tan(splits[i].fov / 2.0f) * splits[i].far_plane;
		float far_width = far_height * splits[i].ratio;

		splits[i].corners[0] = nc - up * near_height - right * near_width; // near-bottom-left
		splits[i].corners[1] = nc + up * near_height - right * near_width; // near-top-left
		splits[i].corners[2] = nc + up * near_height + right * near_width; // near-top-right
		splits[i].corners[3] = nc - up * near_height + right * near_width; // near-bottom-right

		splits[i].corners[4] = fc - up * far_height - right * far_width; // far-bottom-left
		splits[i].corners[5] = fc + up * far_height - right * far_width; // far-top-left
		splits[i].corners[6] = fc + up * far_height + right * far_width; // far-top-right
		splits[i].corners[7] = fc - up * far_height + right * far_width; // far-bottom-right
	}
}

void update_texture_matrices()
{
    for (int i = 0; i < num_cascades; i++)
		texture_matrices[i] = u_Bias * crop[i];
}

void update_far_bounds()
{
    // for every active split
    for(int i = 0 ; i < num_cascades; i++)
    {
        // f[i].fard is originally in eye space - tell's us how far we can see.
        // Here we compute it in camera homogeneous coordinates. Basically, we calculate
        // cam_proj * (0, 0, f[i].fard, 1)^t and then normalize to [0; 1]
        
		vec4 pos = projection * vec4(0.0, 0.0, -splits[i].far_plane, 1.0);
		vec4 ndc = pos / pos.w;

        far_bounds[i] = ndc.z * 0.5 + 0.5;
    }
}

void update_crop_matrices()
{
	mat4 t_projection;

	for (int i = 0; i < num_cascades; i++) 
	{
		vec3 tmax = vec3(-kINFINITY, -kINFINITY, -kINFINITY);
		vec3 tmin = vec3(kINFINITY, kINFINITY, kINFINITY);

		// find the z-range of the current frustum as seen from the light
		// in order to increase precision

		// note that only the z-component is need and thus
		// the multiplication can be simplified
		// transf.z = shad_modelview[2] * f.point[0].x + shad_modelview[6] * f.point[0].y + shad_modelview[10] * f.point[0].z + shad_modelview[14];
		vec4 t_transf = u_ModelView * vec4(splits[i].corners[0], 1.0);

		tmin.z = t_transf.z;
		tmax.z = t_transf.z;

		for (int j = 1; j < 8; j++) 
		{
			t_transf = u_ModelView * vec4(splits[i].corners[j], 1.0);
			if (t_transf.z > tmax.z) 
			{ 
				tmax.z = t_transf.z; 
			}
			if (t_transf.z < tmin.z) 
			{ 
				tmin.z = t_transf.z; 
			}
		}

		//tmax.z += 50; // TODO: This solves the dissapearing shadow problem. but how to fix?

		// Calculate frustum split center
		splits[i].center = vec3(0.0, 0.0, 0.0);

		for (int j = 0; j < 8; j++)
			splits[i].center += splits[i].corners[j];

		splits[i].center /= 8.0;

		if (u_StablePSSM == 1)
		{
			// Calculate bounding sphere radius
			float radius = 0.0;

			for (int j = 0; j < 8; j++)
			{
				float l = length(splits[i].corners[j] - splits[i].center);
				radius = max(radius, l);
			}

			radius = ceil(radius * 16.0) / 16.0;

			// Find bounding box that fits the sphere
			vec3 radius3 = vec3(radius, radius, radius);

			vec3 max = radius3;
			vec3 min = -radius3;

			vec3 cascade_extents = max - min;

			// Push the light position back along the light direction by the near offset.
			vec3 shadow_camera_pos = splits[i].center - direction.xyz * u_NearOffset;

			// Add the near offset to the Z value of the cascade extents to make sure the orthographic frustum captures the entire frustum split (else it will exhibit cut-off issues).
			mat4 ortho = ortho(min.x, max.x, min.y, max.y, -u_NearOffset, u_NearOffset + cascade_extents.z);
			mat4 view = look_at(shadow_camera_pos, splits[i].center, u_CameraUp);

			proj_matrices[i] = ortho;
			crop[i] = ortho * view;

			vec4 shadow_origin = vec4(0.0, 0.0, 0.0, 1.0);
			shadow_origin = crop[i] * shadow_origin;
			shadow_origin = shadow_origin * (u_ShadowMapSize / 2.0);

			vec4 rounded_origin = round(shadow_origin);
			vec4 round_offset = rounded_origin - shadow_origin;
			round_offset = round_offset * (2.0 / u_ShadowMapSize);
			round_offset.z = 0.0;
			round_offset.w = 0.0;

			mat4 shadow_proj = proj_matrices[i];

			shadow_proj[3][0] += round_offset.x;
			shadow_proj[3][1] += round_offset.y;
			shadow_proj[3][2] += round_offset.z;
			shadow_proj[3][3] += round_offset.w;

			crop[i] = shadow_proj * view;
		}
		else
		{
			mat4 t_ortho = ortho(-1.0, 1.0, -1.0, 1.0, -u_NearOffset, -tmin.z);
			mat4 t_shad_mvp = t_ortho * u_ModelView;

			// find the extends of the frustum slice as projected in light's homogeneous coordinates
			for (int j = 0; j < 8; j++)
			{
				t_transf = t_shad_mvp * vec4(splits[i].corners[j], 1.0);

				t_transf.x /= t_transf.w;
				t_transf.y /= t_transf.w;

				if (t_transf.x > tmax.x) { tmax.x = t_transf.x; }
				if (t_transf.x < tmin.x) { tmin.x = t_transf.x; }
				if (t_transf.y > tmax.y) { tmax.y = t_transf.y; }
				if (t_transf.y < tmin.y) { tmin.y = t_transf.y; }
			}

			vec2 tscale = vec2(2.0 / (tmax.x - tmin.x), 2.0 / (tmax.y - tmin.y));
			vec2 toffset = vec2(-0.5 * (tmax.x + tmin.x) * tscale.x, -0.5 * (tmax.y + tmin.y) * tscale.y);

			mat4 t_shad_crop = mat4(1.0);
			t_shad_crop[0][0] = tscale.x;
			t_shad_crop[1][1] = tscale.y;
			t_shad_crop[0][3] = toffset.x;
			t_shad_crop[1][3] = toffset.y;
			t_shad_crop = transpose(t_shad_crop);

			t_projection = t_shad_crop * t_ortho;

			// Store the projection matrix
			proj_matrices[i] = t_projection;
			crop[i] = t_projection * u_ModelView;
		}
	}
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
	for (int i = 0; i < num_cascades; i++)
	{
		splits[i].fov = u_FOV;
		splits[i].ratio = u_Ratio;
	}

	update_splits();
	update_frustum_corners();
	update_crop_matrices();
    update_texture_matrices();
    update_far_bounds();
}

// ------------------------------------------------------------------