// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0, rgba32f) uniform image2D i_Depth;

// ------------------------------------------------------------------
// UNIFORMS ---------------------------------------------------------
// ------------------------------------------------------------------

layout(std430, binding = 0) buffer GlobalUniforms
{
    mat4 view;
    mat4 projection;
    mat4 crop;
};

layout(std140, binding = 1) buffer CSMUniforms
{
    vec4 direction;
    vec4 options;
    int num_cascades;
    float far_bounds[8];
    mat4 texture_matrices[8];
};

uniform float u_Near;
uniform float u_Far;
uniform float u_Lambda;
uniform int u_MaxMip;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

mat4 lookat(vec3 _eye, vec3 _origin, vec3 _up)
{
	vec3 front = normalize(_origin - _eye);
	vec3 right = normalize(cross(front, _up));
	vec3 up = normalize(cross(right, front));

	mat4 m;

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
	mat4 m;

	m[0][0] = 2.0f / (_r - _l);
	m[1][1] = 2.0f / (_t - _b);
	m[3][3] = -2.0f / (_f - _n);

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
    vec2 min_max = textureLod(i_Depth, vec2(0.0), u_MaxMip);

	float nd = depth_exp_to_view(min_max.x);
	float fd = depth_exp_to_view(min_max.y);

	float lambda = u_Lambda;
	float ratio = fd / nd;
	m_splits[0].near_plane = nd;

	for (int i = 1; i < num_cascades; i++)
	{
		float si = i / (float)num_cascades;

		// Practical Split Scheme: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
		float t_near = lambda * (nd * pow(ratio, si)) + (1 - lambda) * (nd + (fd - nd) * si);
		float t_far = t_near * 1.005;
		m_splits[i].near_plane = t_near;
		m_splits[i - 1].far_plane = t_far;
	}

	m_splits[m_split_count - 1].far_plane = fd;
}

void CSM::update_frustum_corners(dw::Camera* camera)
{
	glm::vec3 center = camera->m_position;
	glm::vec3 view_dir = camera->m_forward;

	glm::vec3 up(0.0f, 1.0f, 0.0f);
	glm::vec3 right = glm::cross(view_dir, up);

	for (int i = 0; i < m_split_count; i++)
	{
		FrustumSplit& t_frustum = m_splits[i];

		glm::vec3 fc = center + view_dir * t_frustum.far_plane;
		glm::vec3 nc = center + view_dir * t_frustum.near_plane;

		right = glm::normalize(right);
		up = glm::normalize(glm::cross(right, view_dir));

		// these heights and widths are half the heights and widths of
		// the near and far plane rectangles
		float near_height = tan(t_frustum.fov / 2.0f) * t_frustum.near_plane;
		float near_width = near_height * t_frustum.ratio;
		float far_height = tan(t_frustum.fov / 2.0f) * t_frustum.far_plane;
		float far_width = far_height * t_frustum.ratio;

		t_frustum.corners[0] = nc - up * near_height - right * near_width; // near-bottom-left
		t_frustum.corners[1] = nc + up * near_height - right * near_width; // near-top-left
		t_frustum.corners[2] = nc + up * near_height + right * near_width; // near-top-right
		t_frustum.corners[3] = nc - up * near_height + right * near_width; // near-bottom-right

		t_frustum.corners[4] = fc - up * far_height - right * far_width; // far-bottom-left
		t_frustum.corners[5] = fc + up * far_height - right * far_width; // far-top-left
		t_frustum.corners[6] = fc + up * far_height + right * far_width; // far-top-right
		t_frustum.corners[7] = fc - up * far_height + right * far_width; // far-bottom-right
	}
}

void CSM::update_texture_matrices(dw::Camera* camera)
{
    for (int i = 0; i < m_split_count; i++)
        m_texture_matrices[i] = m_bias * m_crop_matrices[i];
}

void CSM::update_far_bounds(dw::Camera* camera)
{
    // for every active split
    for(int i = 0 ; i < num_cascades ; i++)
    {
        // f[i].fard is originally in eye space - tell's us how far we can see.
        // Here we compute it in camera homogeneous coordinates. Basically, we calculate
        // cam_proj * (0, 0, f[i].fard, 1)^t and then normalize to [0; 1]
        
        FrustumSplit& split = m_splits[i];
		glm::vec4 pos = camera->m_projection * glm::vec4(0.0f, 0.0f, -split.far_plane, 1.0f);
		glm::vec4 ndc = pos / pos.w;

        m_far_bounds[i] = ndc.z * 0.5f + 0.5f;
    }
}

void CSM::update_crop_matrices(glm::mat4 t_modelview, dw::Camera* camera)
{
	glm::mat4 t_projection;
	for (int i = 0; i < m_split_count; i++) 
	{
		FrustumSplit& t_frustum = m_splits[i];

		glm::vec3 tmax(-INFINITY, -INFINITY, -INFINITY);
		glm::vec3 tmin(INFINITY, INFINITY, INFINITY);

		// find the z-range of the current frustum as seen from the light
		// in order to increase precision

		// note that only the z-component is need and thus
		// the multiplication can be simplified
		// transf.z = shad_modelview[2] * f.point[0].x + shad_modelview[6] * f.point[0].y + shad_modelview[10] * f.point[0].z + shad_modelview[14];
		glm::vec4 t_transf = t_modelview * glm::vec4(t_frustum.corners[0], 1.0f);

		tmin.z = t_transf.z;
		tmax.z = t_transf.z;
		for (int j = 1; j < 8; j++) 
		{
			t_transf = t_modelview * glm::vec4(t_frustum.corners[j], 1.0f);
			if (t_transf.z > tmax.z) { tmax.z = t_transf.z; }
			if (t_transf.z < tmin.z) { tmin.z = t_transf.z; }
		}

		//tmax.z += 50; // TODO: This solves the dissapearing shadow problem. but how to fix?

		// Calculate frustum split center
		t_frustum.center = glm::vec3(0.0f, 0.0f, 0.0f);

		for (int j = 0; j < 8; j++)
			t_frustum.center += t_frustum.corners[j];

		t_frustum.center /= 8.0f;

		if (m_stable_pssm)
		{
			// Calculate bounding sphere radius
			float radius = 0.0f;

			for (int j = 0; j < 8; j++)
			{
				float length = glm::length(t_frustum.corners[j] - t_frustum.center);
				radius = glm::max(radius, length);
			}

			radius = ceil(radius * 16.0f) / 16.0f;

			// Find bounding box that fits the sphere
			glm::vec3 radius3(radius, radius, radius);

			glm::vec3 max = radius3;
			glm::vec3 min = -radius3;

			glm::vec3 cascade_extents = max - min;

			// Push the light position back along the light direction by the near offset.
			glm::vec3 shadow_camera_pos = t_frustum.center - m_light_direction * m_near_offset;

			// Add the near offset to the Z value of the cascade extents to make sure the orthographic frustum captures the entire frustum split (else it will exhibit cut-off issues).
			glm::mat4 ortho = glm::ortho(min.x, max.x, min.y, max.y, -m_near_offset, m_near_offset + cascade_extents.z);
			glm::mat4 view = glm::lookAt(shadow_camera_pos, t_frustum.center, camera->m_up);

			m_proj_matrices[i] = ortho;
			m_crop_matrices[i] = ortho * view;

			glm::vec4 shadow_origin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
			shadow_origin = m_crop_matrices[i] * shadow_origin;
			shadow_origin = shadow_origin * (m_shadow_map_size / 2.0f);

			glm::vec4 rounded_origin = glm::round(shadow_origin);
			glm::vec4 round_offset = rounded_origin - shadow_origin;
			round_offset = round_offset * (2.0f / m_shadow_map_size);
			round_offset.z = 0.0f;
			round_offset.w = 0.0f;

			glm::mat4& shadow_proj = m_proj_matrices[i];

			shadow_proj[3][0] += round_offset.x;
			shadow_proj[3][1] += round_offset.y;
			shadow_proj[3][2] += round_offset.z;
			shadow_proj[3][3] += round_offset.w;

			m_crop_matrices[i] = shadow_proj * view;
		}
		else
		{
			glm::mat4 t_ortho = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -m_near_offset, -tmin.z);
			glm::mat4 t_shad_mvp = t_ortho * t_modelview;

			// find the extends of the frustum slice as projected in light's homogeneous coordinates
			for (int j = 0; j < 8; j++)
			{
				t_transf = t_shad_mvp * glm::vec4(t_frustum.corners[j], 1.0f);

				t_transf.x /= t_transf.w;
				t_transf.y /= t_transf.w;

				if (t_transf.x > tmax.x) { tmax.x = t_transf.x; }
				if (t_transf.x < tmin.x) { tmin.x = t_transf.x; }
				if (t_transf.y > tmax.y) { tmax.y = t_transf.y; }
				if (t_transf.y < tmin.y) { tmin.y = t_transf.y; }
			}

			glm::vec2 tscale(2.0f / (tmax.x - tmin.x), 2.0f / (tmax.y - tmin.y));
			glm::vec2 toffset(-0.5f * (tmax.x + tmin.x) * tscale.x, -0.5f * (tmax.y + tmin.y) * tscale.y);

			glm::mat4 t_shad_crop = glm::mat4(1.0f);
			t_shad_crop[0][0] = tscale.x;
			t_shad_crop[1][1] = tscale.y;
			t_shad_crop[0][3] = toffset.x;
			t_shad_crop[1][3] = toffset.y;
			t_shad_crop = glm::transpose(t_shad_crop);

			t_projection = t_shad_crop * t_ortho;

			// Store the projection matrix
			m_proj_matrices[i] = t_projection;
			m_crop_matrices[i] = t_projection * t_modelview;
		}
	}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{

}

// ------------------------------------------------------------------