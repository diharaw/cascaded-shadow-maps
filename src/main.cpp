#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include <algorithm>
#include <memory>
#include <ogl.h>
#include "csm.h"

#undef min
#undef max

// Uniform buffer data structure.
struct ObjectUniforms
{
	DW_ALIGNED(16) glm::mat4 model;
};

struct GlobalUniforms
{
    DW_ALIGNED(16) glm::mat4 view;
    DW_ALIGNED(16) glm::mat4 projection;
    DW_ALIGNED(16) glm::mat4 crop[8];
};

// Structure containing frustum split far-bound.
// 16-byte aligned to adhere to the GLSL std140 memory layout's array packing scheme.
// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout
struct FarBound
{
    DW_ALIGNED(16) float far_bound;
};

struct CSMUniforms
{
	DW_ALIGNED(16) glm::mat4 texture_matrices[8];
    DW_ALIGNED(16) glm::vec4 direction;
    DW_ALIGNED(16) glm::vec4 options; // x: shadows enabled, y: show cascades, z: blend enabled
    DW_ALIGNED(16) int       num_cascades;
    DW_ALIGNED(16) FarBound  far_bounds[8];
};

#define CAMERA_FAR_PLANE 1000.0f
#define CSM_MAX_CASCADES 32

enum CSMSplitMethod
{
    CSM_SPLIT_MANUAL,
    CSM_SPLIT_PSSM,
    CSM_SPLIT_LOGARITHMIC
};

struct CSMDesc
{
    // Clip plane distances of the main camera
    float near_clip;
    float far_clip;
    // Min-Max distances of the cascades [0.0 - 1.0]
    float min_distance;
    float max_distance;
    int num_cascades;
    int shadow_map_size;
    int filter_size;
    CSMSplitMethod split_method;
    float pssm_lambda;
    glm::vec3 light_dir;
    bool stablize_cascades;
    float split_ratios[CSM_MAX_CASCADES];
    float cascade_splits[CSM_MAX_CASCADES];
    glm::vec4 cascade_scales[CSM_MAX_CASCADES];
    glm::vec4 cascade_offsets[CSM_MAX_CASCADES];
};

glm::mat4 make_shadow_matrix(glm::mat4 inverse_vp, glm::vec3 cam_up_dir, glm::vec3 light_dir, bool stablize_cascades)
{
    // Get the 8 points of the view frustum in world space
    glm::vec4 frustum_corners[8] =
    {
        glm::vec4(-1.0f,  1.0f, -1.0f, 1.0f),
        glm::vec4(1.0f,  1.0f, -1.0f, 1.0f),
        glm::vec4(1.0f, -1.0f, -1.0f, 1.0f),
        glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f),
        glm::vec4(-1.0f,  1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f,  1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, -1.0f, 1.0f, 1.0f),
        glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f),
    };

    glm::vec3 frustum_center = glm::vec3(0.0f);

    for (int i = 0; i < 8; i++)
    {
        glm::vec4 c = inverse_vp * frustum_corners[i];
        frustum_corners[i] = c / frustum_corners[i].w;

        frustum_center += frustum_corners[i];
    }

    frustum_center /= 8.0f;

    glm::vec3 up_dir = cam_up_dir;

    if (stablize_cascades)
        up_dir = glm::vec3(0.0f, 1.0f, 1.0f);

    // Get position of the shadow camera
    glm::vec3 shadow_camera_pos = frustum_center + light_dir * -0.5f;

    // Create shadow matrix
    glm::mat4 shadow_view = glm::lookAt(shadow_camera_pos, frustum_center, up_dir);
    glm::mat4 shadow_proj = glm::ortho(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f, 1.0f);

    // Create bias matrix
    glm::mat4 tex_scale_bias = glm::mat4(1.0f);
    tex_scale_bias = glm::scale(tex_scale_bias, glm::vec3(0.5f, 0.5f, 1.0f));
    tex_scale_bias = glm::translate(tex_scale_bias, glm::vec3(0.5f, 0.5f, 0.0f));

    return tex_scale_bias * shadow_proj * shadow_view;
}

void generate_cascade_data(dw::Camera* camera, CSMDesc& desc)
{
    float min_distance = desc.min_distance;
    float max_distance = desc.max_distance;

    float shadow_map_size = static_cast<float>(desc.shadow_map_size);

    float cascade_splits[CSM_MAX_CASCADES];

    if (desc.split_method == CSM_SPLIT_MANUAL)
    {
        for (int i = 0; i < desc.num_cascades; i++)
            cascade_splits[i] = min_distance + desc.split_ratios[i] * max_distance;
    }
    else if (desc.split_method == CSM_SPLIT_PSSM || desc.split_method == CSM_SPLIT_LOGARITHMIC)
    {
        float lambda = 1.0f;

        if (desc.split_method == CSM_SPLIT_PSSM)
            lambda = desc.pssm_lambda;

        float clip_range = desc.near_clip - desc.far_clip;
        float min_z = desc.near_clip + min_distance * clip_range;
        float max_z = desc.near_clip + max_distance * clip_range;
        float range = max_z - min_z;
        float ratio = max_z / min_z;

        for (int i = 0; i < desc.num_cascades; i++)
        {
            float p = (i + 1) / static_cast<float>(desc.num_cascades);
            float log = min_z * pow(ratio, p);
            float uniform = min_z + range * p;
            float d = lambda * (log - uniform) + uniform;
            cascade_splits[i] = (d - desc.near_clip) / clip_range;
        }

        glm::mat4 global_shadow_matrix = make_shadow_matrix(glm::inverse(camera->m_view_projection), camera->m_up, desc.light_dir, desc.stablize_cascades);
        glm::mat4 inv_view_proj = glm::inverse(camera->m_view_projection);

        for (int cascade_idx = 0; cascade_idx < desc.num_cascades; cascade_idx++)
        {
            glm::vec3 frustum_corners[8] =
            {
                glm::vec3(-1.0f,  1.0f, -1.0f),
                glm::vec3(1.0f,  1.0f, -1.0f),
                glm::vec3(1.0f, -1.0f, -1.0f),
                glm::vec3(-1.0f, -1.0f, -1.0f),
                glm::vec3(-1.0f,  1.0f, 1.0f),
                glm::vec3(1.0f,  1.0f, 1.0f),
                glm::vec3(1.0f, -1.0f, 1.0f),
                glm::vec3(-1.0f, -1.0f, 1.0f),
            };

            float prev_split_dist = cascade_idx == 0 ? desc.min_distance : cascade_splits[cascade_idx - 1];
            float split_dist = cascade_splits[cascade_idx];

            for (uint32_t i = 0; i < 8; ++i)
                frustum_corners[i] = glm::vec3(inv_view_proj * glm::vec4(frustum_corners[i], 1.0f));

                // Get the corners of the current cascade slice of the view frustum
                for (uint32_t i = 0; i < 4; ++i)
                {
                    glm::vec3 corner_ray = frustum_corners[i + 4] - frustum_corners[i];
                    glm::vec3 near_corner_ray = corner_ray * prev_split_dist;
                    glm::vec3 far_corner_ray = corner_ray * split_dist;
                    frustum_corners[i + 4] = frustum_corners[i] + far_corner_ray;
                    frustum_corners[i] = frustum_corners[i] + near_corner_ray;
                }

                // Calculate the centroid of the view frustum slice
                glm::vec3 frustum_center = glm::vec3(0.0f);

                for (uint32_t i = 0; i < 8; ++i)
                    frustum_center = frustum_center + frustum_corners[i];

                frustum_center *= 1.0f / 8.0f;

                // Pick the up vector to use for the light camera
                glm::vec3 up_dir = camera->m_right;

                glm::vec3 min_extents;
                glm::vec3 max_extents;

                if (desc.stablize_cascades)
                {
                    // This needs to be constant for it to be stable
                    up_dir = glm::vec3(0.0f, 1.0f, 0.0f);

                    // Calculate the radius of a bounding sphere surrounding the frustum corners
                    float sphere_radius = 0.0f;
                    for (uint32_t i = 0; i < 8; ++i)
                    {
                        float dist = glm::length(glm::vec3(frustum_corners[i]) - frustum_center);
                        sphere_radius = std::max(sphere_radius, dist);
                    }

                    sphere_radius = std::ceil(sphere_radius * 16.0f) / 16.0f;

                    max_extents = glm::vec3(sphere_radius, sphere_radius, sphere_radius);
                    min_extents = -max_extents;
                }
                else
                {
                    // Create a temporary view matrix for the light
                    glm::vec3 light_camera_pos = frustum_center;
                    glm::vec3 look_at = frustum_center - desc.light_dir;
                    glm::mat4 light_view = glm::lookAt(light_camera_pos, look_at, up_dir);

                    // Calculate an AABB around the frustum corners
                    glm::vec3 mins = glm::vec3(FLT_MAX);
                    glm::vec3 maxes = glm::vec3(-FLT_MAX);

                    for (uint32_t i = 0; i < 8; ++i)
                    {
                        glm::vec3 corner = glm::vec3(light_view * glm::vec4(frustum_corners[i], 1.0f));
                        mins = glm::min(mins, corner);
                        maxes = glm::min(maxes, corner);
                    }

                    min_extents = mins;
                    max_extents = maxes;

                    // Adjust the min/max to accommodate the filtering size
                    float scale = (desc.shadow_map_size + desc.filter_size) / static_cast<float>(desc.shadow_map_size);
                    min_extents.x *= scale;
                    min_extents.y *= scale;
                    max_extents.x *= scale;
                    max_extents.y *= scale;
                }

                glm::vec3 cascade_extents = max_extents - min_extents;

                // Get position of the shadow camera
                glm::vec3 shadow_camera_pos = frustum_center + desc.light_dir * -min_extents.z;

                // Come up with a new orthographic camera for the shadow caster
                glm::mat4 shadow_proj = glm::ortho(min_extents.x, min_extents.y, max_extents.x, max_extents.y, 0.0f, cascade_extents.z);
                glm::mat4 shadow_view = glm::lookAt(shadow_camera_pos, frustum_center, up_dir);
                glm::mat4 shadow_matrix = shadow_proj * shadow_view;

                if (desc.stablize_cascades)
                {
                    // Create the rounding matrix, by projecting the world-space origin and determining
                    // the fractional offset in texel space
                    glm::vec4 shadow_origin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    shadow_origin = shadow_matrix * shadow_origin;
                    shadow_origin = shadow_origin * (desc.shadow_map_size / 2.0f);

                    glm::vec4 rounded_origin = glm::round(shadow_origin);
                    glm::vec4 round_offset = rounded_origin - shadow_origin;
                    round_offset = round_offset * (2.0f / desc.shadow_map_size);
                    round_offset.z = 0.0f;
                    round_offset.w = 0.0f;

                    shadow_proj[3] = shadow_proj[3] + round_offset;
                }

                // Draw the mesh with depth only, using the new shadow camera
                // RenderDepthCPU(context, shadowCamera, world, characterWorld, true);

                // Apply the scale/offset matrix, which transforms from [-1,1]
                // post-projection space to [0,1] UV space
                glm::mat4 tex_scale_bias;
                tex_scale_bias[0] = glm::vec4(0.5f, 0.0f, 0.0f, 0.0f);
                tex_scale_bias[1] = glm::vec4(0.0f, -0.5f, 0.0f, 0.0f);
                tex_scale_bias[2] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
                tex_scale_bias[3] = glm::vec4(0.5f, 0.5f, 0.0f, 1.0f);

                shadow_matrix = tex_scale_bias * shadow_matrix;

                // Store the split distance in terms of view space depth
                const float clip_dist = desc.far_clip - desc.near_clip;

                desc.cascade_splits[cascade_idx] = desc.near_clip + split_dist * clip_dist;

                // Calculate the position of the lower corner of the cascade partition, in the UV space
                // of the first cascade partition
                glm::mat4 inv_cascade_mat = glm::inverse(shadow_matrix);
                glm::vec3 cascade_corner = glm::vec3(inv_cascade_mat * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
                cascade_corner = global_shadow_matrix * glm::vec4(cascade_corner, 1.0f);

                // Do the same for the upper corner
                glm::vec3 other_corner = glm::vec3(inv_cascade_mat * glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
                other_corner = glm::vec3(global_shadow_matrix * glm::vec4(other_corner, 1.0f));

                // Calculate the scale and offset
                glm::vec3 cascade_scale = glm::vec3(1.0f, 1.0f, 1.0f) / (other_corner - cascade_corner);

                desc.cascade_offsets[cascade_idx] = glm::vec4(-cascade_corner, 0.0f);
                desc.cascade_scales[cascade_idx] = glm::vec4(cascade_scale, 1.0f);
        }
    }
}


class Sample : public dw::Application
{
protected:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	bool init(int argc, const char* argv[]) override
	{
		// Create GPU resources.
		/*if (!create_shaders())
			return false;

		if (!create_uniform_buffer())
			return false;

		create_framebuffers();*/

		// Load mesh.
		if (!load_mesh())
			return false;

		// Create camera.
		create_camera();

		// Initial CSM.
		//initialize_csm();
		
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update(double delta) override
	{
        // Debug GUI
        //debug_gui();
        
		// Update camera.
        update_camera();
        
        // Update CSM.
		//if (!m_ssdm)
		//	m_csm.update(m_main_camera.get(), m_csm_uniforms.direction);
  //  
		//// Update transforms.
  //      update_transforms(m_debug_mode ? m_debug_camera.get() : m_main_camera.get());

		//// Update global uniforms.
		//update_global_uniforms(m_global_uniforms);

		//// Update CSM uniforms.
		//update_csm_uniforms(m_csm_uniforms);

		//if (m_ssdm)
		//{
		//	// Render depth prepass
		//	render_depth_prepass();

		//	copy_depth();

		//	depth_reduction();

		//	setup_cascades_sdsm();
		//}

  //      // Render debug view.
  //      render_debug_view();
  //      
  //      // Render shadow map.
  //      render_shadow_map();
  //      
  //      // Render scene.
  //      render_scene();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, m_width, m_height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (m_debug_mode)
			m_debug_draw.frustum(m_main_camera->m_view_projection, glm::vec3(0.0f, 1.0f, 0.0f));
        
        // Render debug draw.
        m_debug_draw.render(nullptr, m_width, m_height, m_debug_mode ? m_debug_camera->m_view_projection : m_main_camera->m_view_projection);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void shutdown() override
	{
		m_depth_prepass_fbo.reset();
		m_depth_prepass_rt.reset();

		// Cleanup CSM.
		m_csm.shutdown();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::AppSettings intial_app_settings()
	{
		dw::AppSettings settings;

		settings.width = 1920;
		settings.height = 1080;
		settings.title = "Cascaded Shadow Maps";

		return settings;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void create_framebuffers()
	{
		for (auto& fbo : m_depth_reduction_fbos)
			fbo.reset();

		m_depth_reduction_fbos.clear();

		m_depth_reduction_rt.reset();

		m_depth_prepass_fbo.reset();
		m_depth_prepass_rt.reset();
		
		m_depth_prepass_rt = std::make_unique<dw::gl::Texture2D>(m_width, m_height, 1, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
		m_depth_prepass_rt->set_min_filter(GL_LINEAR);
		m_depth_prepass_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
		
		m_depth_prepass_fbo = std::make_unique<dw::gl::Framebuffer>();

		m_depth_prepass_fbo->attach_depth_stencil_target(m_depth_prepass_rt.get(), 0, 0);

		int32_t w = m_width;
		int32_t h = m_height;
		int32_t count = 0;

		while (!(w == 1 && h == 1))
		{
			count++;
			w /= 2;
			h /= 2;

			w = std::max(w, 1);
			h = std::max(h, 1);
		}

		m_depth_mips = count;
		m_depth_reduction_fbos.resize(count + 1);

		m_depth_reduction_rt = std::make_unique<dw::gl::Texture2D>(m_width, m_height, 1, count, 1, GL_RG32F, GL_RG, GL_FLOAT);
		m_depth_reduction_rt->generate_mipmaps();
		m_depth_reduction_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		for (int i = 0; i <= count; i++)
		{
			m_depth_reduction_fbos[i] = std::make_unique<dw::gl::Framebuffer>();
			m_depth_reduction_fbos[i]->attach_render_target(0, m_depth_reduction_rt.get(), 0, i);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void window_resized(int width, int height) override
	{
		// Override window resized method to update camera projection.
		m_main_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height));
        m_debug_camera->update_projection(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height));

		// Re-initialize CSM to fit new frustum shape.
		initialize_csm();

		create_framebuffers();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
    
    void key_pressed(int code) override
    {
        // Handle forward movement.
        if(code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if(code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;
        
        // Handle sideways movement.
        if(code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if(code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

		if (code == GLFW_KEY_G)
			m_debug_mode = !m_debug_mode;
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void key_released(int code) override
    {
        // Handle forward movement.
        if(code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;
        
        // Handle sideways movement.
        if(code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------

	void mouse_pressed(int code) override
	{
		// Enable mouse look.
		if (code == GLFW_MOUSE_BUTTON_LEFT)
			m_mouse_look = true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void mouse_released(int code) override
	{
		// Disable mouse look.
		if (code == GLFW_MOUSE_BUTTON_LEFT)
			m_mouse_look = false;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

private:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	void initialize_csm()
	{
		m_csm_uniforms.direction = glm::vec4(glm::vec3(m_light_dir_x, -1.0f, m_light_dir_z), 0.0f);
		m_csm_uniforms.direction = glm::normalize(m_csm_uniforms.direction);
        m_csm_uniforms.options.x = 1;
        m_csm_uniforms.options.y = 0;
        m_csm_uniforms.options.z = 1;

		m_csm.initialize(m_pssm_lambda, m_near_offset, m_cascade_count, m_shadow_map_size, m_main_camera.get(), m_width, m_height, m_csm_uniforms.direction);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_shaders()
	{
		// Create general shaders
        m_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/scene_vs.glsl"));
		m_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/scene_fs.glsl"));

		if (!m_vs || !m_fs)
		{
			DW_LOG_FATAL("Failed to create Shaders");
			return false;
		}

		// Create general shader program
        dw::gl::Shader* shaders[] = { m_vs.get(), m_fs.get() };
        m_program = std::make_unique<dw::gl::Program>(2, shaders);

		if (!m_program)
		{
			DW_LOG_FATAL("Failed to create Shader Program");
			return false;
		}
        
        m_program->uniform_block_binding("ObjectUniforms", 1);
        
        // Create CSM shaders
        m_csm_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/shadow_map_vs.glsl"));
        m_csm_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/empty_fs.glsl"));
        
        if (!m_csm_vs || !m_csm_fs)
        {
            DW_LOG_FATAL("Failed to create CSM Shaders");
            return false;
        }
        
        // Create CSM shader program
        dw::gl::Shader* csm_shaders[] = { m_csm_vs.get(), m_csm_fs.get() };
        m_csm_program = std::make_unique<dw::gl::Program>(2, csm_shaders);
        
        if (!m_csm_program)
        {
            DW_LOG_FATAL("Failed to create CSM Shader Program");
            return false;
        }
        
        m_csm_program->uniform_block_binding("ObjectUniforms", 1);

		// Create depth prepass shaders
		m_depth_prepass_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/depth_prepass_vs.glsl"));
		m_depth_prepass_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/empty_fs.glsl"));

		if (!m_depth_prepass_vs || !m_depth_prepass_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Prepass Shaders");
			return false;
		}

		// Create Depth prepass shader program
		dw::gl::Shader* depth_prepass_shaders[] = { m_depth_prepass_vs.get(), m_depth_prepass_fs.get() };
		m_depth_prepass_program = std::make_unique<dw::gl::Program>(2, depth_prepass_shaders);

		if (!m_depth_prepass_program)
		{
			DW_LOG_FATAL("Failed to create Depth Prepass Shader Program");
			return false;
		}

		m_depth_prepass_program->uniform_block_binding("ObjectUniforms", 1);

		// Create depth copy shaders
		m_depth_copy_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/fullscreen_vs.glsl"));
		m_depth_copy_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/copy_fs.glsl"));

		if (!m_depth_copy_vs || !m_depth_copy_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Copy Shaders");
			return false;
		}

		// Create Depth copy shader program
		dw::gl::Shader* depth_copy_shaders[] = { m_depth_copy_vs.get(), m_depth_copy_fs.get() };
		m_depth_copy_program = std::make_unique<dw::gl::Program>(2, depth_copy_shaders);

		if (!m_depth_copy_program)
		{
			DW_LOG_FATAL("Failed to create Depth Copy Shader Program");
			return false;
		}

		// Create depth reduction shaders
		m_depth_reduction_vs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_VERTEX_SHADER, "shader/fullscreen_vs.glsl"));
		m_depth_reduction_fs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_FRAGMENT_SHADER, "shader/depth_reduction_fs.glsl"));

		if (!m_depth_reduction_vs || !m_depth_reduction_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Reduction Shaders");
			return false;
		}

		// Create Depth reduction shader program
		dw::gl::Shader* depth_reduction_shaders[] = { m_depth_reduction_vs.get(), m_depth_reduction_fs.get() };
		m_depth_reduction_program = std::make_unique<dw::gl::Program>(2, depth_reduction_shaders);

		if (!m_depth_reduction_program)
		{
			DW_LOG_FATAL("Failed to create Depth Reduction Shader Program");
			return false;
		}

		// Create setup cascades shader
		m_setup_shadows_cs = std::unique_ptr<dw::gl::Shader>(dw::gl::Shader::create_from_file(GL_COMPUTE_SHADER, "shader/setup_cascades_cs.glsl"));
	
		if (!m_setup_shadows_cs)
		{
			DW_LOG_FATAL("Failed to create Setup Cascades Shaders");
			return false;
		}

		// Create Depth reduction shader program
		dw::gl::Shader* setup_shadows_shaders[] = { m_setup_shadows_cs.get() };
		m_setup_shadows_program = std::make_unique<dw::gl::Program>(1, setup_shadows_shaders);

		if (!m_setup_shadows_program)
		{
			DW_LOG_FATAL("Failed to create Setup Cascades Shader Program");
			return false;
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_uniform_buffer()
	{
		// Create uniform buffer for object matrix data
        m_object_ubo = std::make_unique<dw::gl::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(ObjectUniforms));
        
        // Create uniform buffer for global data
        m_global_ubo = std::make_unique<dw::gl::ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(GlobalUniforms));
        
        // Create uniform buffer for CSM data
        m_csm_ubo = std::make_unique<dw::gl::ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(CSMUniforms));

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool load_mesh()
	{
		//m_plane = dw::Mesh::load("plane.obj", &m_device);
        m_suzanne = dw::Mesh::load("sponza.obj", false);
		return m_suzanne != nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void create_camera()
	{
        m_main_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_debug_camera = std::make_unique<dw::Camera>(60.0f, 0.1f, CAMERA_FAR_PLANE * 2.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

    void render_mesh(dw::Mesh::Ptr mesh, const ObjectUniforms& transforms, bool use_textures = true)
	{
        // Copy new data into UBO.
        update_object_uniforms(transforms);

		// Bind uniform buffers.
        m_global_ubo->bind_base(0);
        m_object_ubo->bind_base(1);

		// Bind vertex array.
        mesh->mesh_vertex_array()->bind();

		for (uint32_t i = 0; i < mesh->sub_mesh_count(); i++)
		{
			dw::SubMesh& submesh = mesh->sub_meshes()[i];

			// Bind texture.
            if (use_textures)
				mesh->material(submesh.mat_idx)->texture(0)->bind(0);

			// Issue draw call.
			glDrawElementsBaseVertex(GL_TRIANGLES, submesh.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * submesh.base_index), submesh.base_vertex);
		}
	}
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void render_scene()
    {
        // Bind and set viewport.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, m_width, m_height);
  
        // Clear default framebuffer.
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Bind states.
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        
        // Bind shader program.
        m_program->use();
        
        // Bind shadow map.
        m_csm.shadow_map()->bind(1);
        
        m_program->set_uniform("s_ShadowMap", 1);

        // Bind uniform buffers.
        m_csm_ubo->bind_base(2);
        
        // Draw meshes.
        //render_mesh(m_plane, m_plane_transforms);
        render_mesh(m_suzanne, m_suzanne_transforms, false);
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	void render_depth_prepass()
	{
		// Bind and set viewport.
		m_depth_prepass_fbo->bind();
		glViewport(0, 0, m_width, m_height);

		// Clear default framebuffer.
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Bind states.
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		// Bind shader program.
		m_depth_prepass_program->use();

		// Draw meshes.
		//render_mesh(m_plane, m_plane_transforms);
		render_mesh(m_suzanne, m_suzanne_transforms, false);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
    
	void copy_depth()
	{
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		m_depth_copy_program->use();

		m_depth_reduction_fbos[0]->bind();
		glViewport(0, 0, m_width, m_height);

		glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m_depth_copy_program->set_uniform("s_Texture", 0);
		m_depth_prepass_rt->bind(0);

		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void depth_reduction()
	{
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		m_depth_reduction_program->use();

		for (uint32_t i = 1; i < m_depth_reduction_fbos.size(); i++)
		{
			float scale = pow(2, i);
			int w = m_width / scale;
			int h = m_height / scale;

			m_depth_reduction_fbos[i]->bind();
			glViewport(0, 0, w, h);
			
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			m_depth_reduction_program->set_uniform("s_Texture", 0);
			m_depth_reduction_rt->bind(0);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, i - 1);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, i - 1);

			glDrawArrays(GL_TRIANGLES, 0, 3);
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
	}

    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void render_shadow_map()
    {
        // Bind states.
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        
        // Bind shader program.
        m_csm_program->use();
        
        for (int i = 0; i < m_csm.frustum_split_count(); i++)
        {
			m_csm_program->set_uniform("u_CascadeIndex", i);

            // Bind and set viewport.
            m_csm.framebuffers()[i]->bind();
            glViewport(0, 0, m_csm.shadow_map_size(), m_csm.shadow_map_size());
            
            // Clear default framebuffer.
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            // Draw meshes. Disable textures because we don't need them here.
			//m_device.bind_rasterizer_state(m_rs);
           // render_mesh(m_plane, m_plane_transforms, false);

            render_mesh(m_suzanne, m_suzanne_transforms, false);
        }
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	void setup_cascades_sdsm()
	{
		m_setup_shadows_program->use();

		m_csm.bind_sdsm_uniforms(m_setup_shadows_program.get(), m_main_camera.get(), m_csm_uniforms.direction);

		m_setup_shadows_program->set_uniform("u_Near", m_main_camera->m_near);
		m_setup_shadows_program->set_uniform("u_Far", m_main_camera->m_far);
		m_setup_shadows_program->set_uniform("u_CameraPos", m_main_camera->m_position);
		m_setup_shadows_program->set_uniform("u_CameraDir", m_main_camera->m_forward);
		m_setup_shadows_program->set_uniform("u_CameraUp", m_main_camera->m_up);
		m_setup_shadows_program->set_uniform("u_MaxMip", m_depth_mips - 1);

		if (m_setup_shadows_program->set_uniform("u_Depth", 0))
			m_depth_reduction_rt->bind(0);

		m_global_ubo->bind_base(0);
		m_csm_ubo->bind_base(1);

		glDispatchCompute(1, 1, 1);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update_object_uniforms(const ObjectUniforms& transform)
	{
        void* ptr = m_object_ubo->map(GL_WRITE_ONLY);
		memcpy(ptr, &transform, sizeof(ObjectUniforms));
        m_object_ubo->unmap();
	}
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_csm_uniforms(const CSMUniforms& csm)
    {
        void* ptr = m_csm_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, &csm, sizeof(CSMUniforms));
        m_csm_ubo->unmap();
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_global_uniforms(const GlobalUniforms& global)
    {
        void* ptr = m_global_ubo->map(GL_WRITE_ONLY);
        memcpy(ptr, &global, sizeof(GlobalUniforms));
        m_global_ubo->unmap();
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_transforms(dw::Camera* camera)
    {
        // Update camera matrices.
        m_global_uniforms.view = camera->m_view;
        m_global_uniforms.projection = camera->m_projection;
        
        // Update CSM farbounds.
        m_csm_uniforms.num_cascades = m_csm.frustum_split_count();
        
        for (int i = 0; i < m_csm.frustum_split_count(); i++)
        {
            m_csm_uniforms.far_bounds[i].far_bound = m_csm.far_bound(i);
            m_csm_uniforms.texture_matrices[i] = m_csm.texture_matrix(i);
			m_global_uniforms.crop[i] = m_csm.split_view_proj(i);
        }
        
        // Update plane transforms.
        m_plane_transforms.model = glm::mat4(1.0f);

        // Update suzanne transforms.
        m_suzanne_transforms.model = glm::mat4(1.0f);
        m_suzanne_transforms.model = glm::translate(m_suzanne_transforms.model, glm::vec3(0.0f, 3.0f, 0.0f));
       // m_suzanne_transforms.model = glm::rotate(m_suzanne_transforms.model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
        m_suzanne_transforms.model = glm::scale(m_suzanne_transforms.model, glm::vec3(0.1f));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_camera()
    {
        dw::Camera* current = m_main_camera.get();
        
        if (m_debug_mode)
            current = m_debug_camera.get();
        
        float forward_delta = m_heading_speed * m_delta;
        float right_delta = m_sideways_speed * m_delta;
        
        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);
        
        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_mouse_delta_y * m_camera_sensitivity),
                                                    (float)(m_mouse_delta_x * m_camera_sensitivity),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }
        
        current->update();
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void debug_gui()
    {
        if (ImGui::Begin("Cascaded Shadow Maps"))
        {
            bool shadows = m_csm_uniforms.options.x;
            ImGui::Checkbox("Enable Shadows", &shadows);
            m_csm_uniforms.options.x = shadows;
            
            bool debug = m_csm_uniforms.options.y;
            ImGui::Checkbox("Show Debug Cascades", &debug);
            m_csm_uniforms.options.y = debug;
            
            bool blend = m_csm_uniforms.options.z;
            ImGui::Checkbox("Blending", &blend);
            m_csm_uniforms.options.z = blend;

			ImGui::Checkbox("Sample Distribution Shadow Maps", &m_ssdm);
			ImGui::Checkbox("Stable", &m_csm.m_stable_pssm);
            ImGui::Checkbox("Debug Camera", &m_debug_mode);
            ImGui::Checkbox("Show Frustum Splits", &m_show_frustum_splits);
            ImGui::Checkbox("Show Cascade Frustum", &m_show_cascade_frustums);
            
            ImGui::SliderFloat("Lambda", &m_csm.m_lambda, 0, 1);
            
            int split_count = m_csm.m_split_count;
            ImGui::SliderInt("Frustum Splits", &split_count, 1, 4);

            if (split_count != m_csm.m_split_count)
                m_csm.initialize(m_csm.m_lambda, m_csm.m_near_offset, split_count, m_csm.m_shadow_map_size, m_main_camera.get(), m_width, m_height, glm::vec3(m_csm_uniforms.direction));
            
			float near_offset = m_csm.m_near_offset;
			ImGui::SliderFloat("Near Offset", &near_offset, 100.0f, 1000.0f);

			if (m_near_offset != near_offset)
			{
				m_near_offset = near_offset;
				m_csm.initialize(m_csm.m_lambda, m_near_offset, split_count, m_csm.m_shadow_map_size, m_main_camera.get(), m_width, m_height, glm::vec3(m_csm_uniforms.direction));
			}

			ImGui::SliderFloat("Light Direction X", &m_light_dir_x, 0.0f, 1.0f);
			ImGui::SliderFloat("Light Direction Z", &m_light_dir_z, 0.0f, 1.0f);

			m_csm_uniforms.direction = glm::vec4(glm::vec3(m_light_dir_x, -1.0f, m_light_dir_z), 0.0f);
			m_csm_uniforms.direction = glm::normalize(m_csm_uniforms.direction);

            static const char* items[] = { "256", "512", "1024", "2048" };
            static const int shadow_map_sizes[] = { 256, 512, 1024, 2048 };
            static int item_current = 3;
            ImGui::Combo("Shadow Map Size", &item_current, items, IM_ARRAYSIZE(items));
            
            if (shadow_map_sizes[item_current] != m_csm.m_shadow_map_size)
                m_csm.initialize(m_csm.m_lambda, m_csm.m_near_offset, m_csm.m_split_count, shadow_map_sizes[item_current], m_main_camera.get(), m_width, m_height, glm::vec3(m_csm_uniforms.direction));
            
            static int current_view = 0;
            ImGui::RadioButton("Scene", &current_view, 0);
            
            for (int i = 0; i < m_csm.m_split_count; i++)
            {
                std::string name = "Cascade " + std::to_string(i + 1);
                ImGui::RadioButton(name.c_str(), &current_view, i + 1);
            }
        }
        ImGui::End();
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void render_debug_view()
    {
        for (int i = 0; i < m_csm.m_split_count; i++)
        {
            FrustumSplit& split = m_csm.frustum_splits()[i];
            
            // Render frustum splits.
            if (m_show_frustum_splits)
            {
                m_debug_draw.line(split.corners[0], split.corners[3], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[3], split.corners[2], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[2], split.corners[1], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[1], split.corners[0], glm::vec3(1.0f));
                
                m_debug_draw.line(split.corners[4], split.corners[7], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[7], split.corners[6], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[6], split.corners[5], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[5], split.corners[4], glm::vec3(1.0f));
                
                m_debug_draw.line(split.corners[0], split.corners[4], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[1], split.corners[5], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[2], split.corners[6], glm::vec3(1.0f));
                m_debug_draw.line(split.corners[3], split.corners[7], glm::vec3(1.0f));
            }
            
            // Render shadow frustums.
            if (m_show_cascade_frustums)
                m_debug_draw.frustum(m_csm.split_view_proj(i), glm::vec3(1.0f, 0.0f, 0.0f));
        }
        
        if (m_debug_mode)
            m_debug_draw.frustum(m_main_camera->m_projection, m_main_camera->m_view, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------

private:
	// Clear color.
	float m_clear_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

	// General GPU resources.
    std::unique_ptr<dw::gl::Shader> m_vs;
	std::unique_ptr<dw::gl::Shader> m_fs;
	std::unique_ptr<dw::gl::Program> m_program;
	std::unique_ptr<dw::gl::UniformBuffer> m_object_ubo;
    std::unique_ptr<dw::gl::ShaderStorageBuffer> m_csm_ubo;
    std::unique_ptr<dw::gl::ShaderStorageBuffer> m_global_ubo;
    
    // CSM shaders.
    std::unique_ptr<dw::gl::Shader> m_csm_vs;
    std::unique_ptr<dw::gl::Shader> m_csm_fs;
    std::unique_ptr<dw::gl::Program> m_csm_program;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    std::unique_ptr<dw::Camera> m_debug_camera;

	// Depth Pre-pass
	std::unique_ptr<dw::gl::Shader> m_depth_prepass_vs;
	std::unique_ptr<dw::gl::Shader> m_depth_prepass_fs;
	std::unique_ptr<dw::gl::Program> m_depth_prepass_program;
	std::unique_ptr<dw::gl::Texture2D> m_depth_prepass_rt;
	std::unique_ptr<dw::gl::Framebuffer> m_depth_prepass_fbo;

	std::unique_ptr<dw::gl::Shader> m_depth_copy_vs;
	std::unique_ptr<dw::gl::Shader> m_depth_copy_fs;
	std::unique_ptr<dw::gl::Program> m_depth_copy_program;

	std::unique_ptr<dw::gl::Shader> m_depth_reduction_vs;
	std::unique_ptr<dw::gl::Shader> m_depth_reduction_fs;
	std::unique_ptr<dw::gl::Program> m_depth_reduction_program;
	std::unique_ptr<dw::gl::Texture2D> m_depth_reduction_rt;
	std::vector<std::unique_ptr<dw::gl::Framebuffer>> m_depth_reduction_fbos;
    
	std::unique_ptr<dw::gl::Shader> m_setup_shadows_cs;
	std::unique_ptr<dw::gl::Program> m_setup_shadows_program;

	// Assets.
	dw::Mesh::Ptr m_plane;
    dw::Mesh::Ptr m_suzanne;

	// Uniforms.
	ObjectUniforms m_plane_transforms;
    ObjectUniforms m_suzanne_transforms;
    GlobalUniforms m_global_uniforms;
    CSMUniforms m_csm_uniforms;

	// Cascaded Shadow Mapping.
	CSM m_csm;
    
    // Camera controls.
    bool m_mouse_look = false;
    bool m_debug_mode = false;
    float m_heading_speed = 0.0f;
    float m_sideways_speed = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed = 0.1f;

	// Default shadow options.
	int m_depth_mips = 0;
	bool m_ssdm = false;
	int m_shadow_map_size = 2048;
	int m_cascade_count = 4;
	float m_pssm_lambda = 0.3;
	float m_near_offset = 250.0f;
	float m_light_dir_x = -1.0f;
	float m_light_dir_z = 0.0f;
    
    // Debug options.
    bool m_show_frustum_splits = false;
    bool m_show_cascade_frustums = false;
};

DW_DECLARE_MAIN(Sample)
