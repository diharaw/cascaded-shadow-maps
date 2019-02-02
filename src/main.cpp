#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include <memory>
#include "csm.h"

// Embedded vertex shader source.
const char* g_sample_vs_src = R"(

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

layout(std430, binding = 0) buffer GlobalUniforms
{
    mat4 view;
    mat4 projection;
    mat4 crop[8];
};

layout (std140) uniform ObjectUniforms //#binding 1
{
    mat4 model;
};

out vec3 PS_IN_WorldFragPos;
out vec4 PS_IN_NDCFragPos;
out vec3 PS_IN_Normal;
out vec2 PS_IN_TexCoord;

void main()
{
    vec4 position = model * vec4(VS_IN_Position, 1.0);
	PS_IN_WorldFragPos = position.xyz;
	PS_IN_Normal = mat3(model) * VS_IN_Normal;
	PS_IN_TexCoord = VS_IN_TexCoord;
    PS_IN_NDCFragPos = projection * view * position;
    gl_Position = PS_IN_NDCFragPos;
}

)";

const char* g_csm_vs_src = R"(

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

layout(std430, binding = 0) buffer GlobalUniforms
{
    mat4 view;
    mat4 projection;
    mat4 crop[8];
};

layout (std140) uniform ObjectUniforms //#binding 1
{
    mat4 model;
};

uniform int u_CascadeIndex;

void main()
{
    gl_Position = crop[u_CascadeIndex] * model * vec4(VS_IN_Position, 1.0);
}

)";

const char* g_depth_prepass_vs_src = R"(

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

layout(std430, binding = 0) buffer GlobalUniforms 
{
    mat4 view;
    mat4 projection;
    mat4 crop[8];
};

layout (std140) uniform ObjectUniforms //#binding 1
{
    mat4 model;
};

void main()
{
    gl_Position = projection * view * model * vec4(VS_IN_Position, 1.0);
}

)";

// Embedded fragment shader source.
const char* g_sample_fs_src = R"(

out vec4 PS_OUT_Color;

in vec3 PS_IN_WorldFragPos;
in vec4 PS_IN_NDCFragPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

layout(std140, binding = 2) buffer CSMUniforms
{
    vec4 direction;
    vec4 options;
    int num_cascades;
    float far_bounds[8];
    mat4 texture_matrices[8];
};

uniform sampler2D s_Diffuse; //#slot 0
uniform sampler2DArray s_ShadowMap; //#slot 1

float depth_compare(float a, float b, float bias)
{
    return a - bias > b ? 1.0 : 0.0;
}

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

)";

const char* g_csm_fs_src = R"(

void main()
{
}

)";

const char* g_post_process_vs_src = R"(

out vec2 PS_IN_TexCoord;

void main()
{
    PS_IN_TexCoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
	gl_Position = vec4(PS_IN_TexCoord * 2.0f + -1.0f, 0.0f, 1.0f);
}

)";

const char* g_depth_reduction_fs_src = R"(

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

)";

const char* g_depth_copy_src = R"(

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
	PS_OUT_Color = texture(s_Texture, PS_IN_TexCoord).xx;
}

// ------------------------------------------------------------------

)";

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
    DW_ALIGNED(16) glm::vec4 direction;
    DW_ALIGNED(16) glm::vec4 options; // x: shadows enabled, y: show cascades, z: blend enabled
    DW_ALIGNED(16) int       num_cascades;
    DW_ALIGNED(16) FarBound  far_bounds[8];
    DW_ALIGNED(16) glm::mat4 texture_matrices[8];
};

#define CAMERA_FAR_PLANE 1000.0f

class Sample : public dw::Application
{
protected:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	bool init(int argc, const char* argv[]) override
	{
		// Create GPU resources.
		if (!create_shaders())
			return false;

		if (!create_uniform_buffer())
			return false;

		create_framebuffers();

		// Load mesh.
		if (!load_mesh())
			return false;

		// Create camera.
		create_camera();

		// Initial CSM.
		initialize_csm();
		
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update(double delta) override
	{
        // Debug GUI
        debug_gui();
        
		// Update camera.
        update_camera();
        
        // Update CSM.
        m_csm.update(m_main_camera.get(), m_csm_uniforms.direction);
    
		// Update transforms.
        update_transforms(m_debug_mode ? m_debug_camera.get() : m_main_camera.get());

		// Update global uniforms.
		update_global_uniforms(m_global_uniforms);

		// Update CSM uniforms.
		update_csm_uniforms(m_csm_uniforms);

		if (m_ssdm)
		{
			// Render depth prepass
			render_depth_prepass();

			copy_depth();

			depth_reduction();

		}

        // Render debug view.
        render_debug_view();
        
        // Render shadow map.
        render_shadow_map();
        
        // Render scene.
        render_scene();
        
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
        
		// Unload assets.
		dw::Mesh::unload(m_plane);
        dw::Mesh::unload(m_suzanne);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	dw::AppSettings intial_app_settings()
	{
		dw::AppSettings settings;

		settings.width = 1280;
		settings.height = 720;
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

		m_depth_prepass_rt = std::make_unique<dw::Texture2D>(m_width, m_height, 1, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
		m_depth_prepass_rt->set_min_filter(GL_LINEAR);
		m_depth_prepass_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
		
		m_depth_prepass_fbo = std::make_unique<dw::Framebuffer>();

		m_depth_prepass_fbo->attach_depth_stencil_target(m_depth_prepass_rt.get(), 0, 0);

		uint32_t w = m_width;
		uint32_t h = m_height;
		uint32_t count = 0;

		while (w > 1 && h > 1)
		{
			count++;
			w /= 2;
			h /= 2;
		}

		m_depth_reduction_fbos.resize(count + 1);

		m_depth_reduction_rt = std::make_unique<dw::Texture2D>(m_width, m_height, 1, count, 1, GL_RG32F, GL_RG, GL_FLOAT);
		m_depth_reduction_rt->generate_mipmaps();
		m_depth_reduction_rt->set_min_filter(GL_LINEAR);
		m_depth_reduction_rt->set_wrapping(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);

		for (int i = 0; i <= count; i++)
		{
			m_depth_reduction_fbos[i] = std::make_unique<dw::Framebuffer>();
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
        m_vs = std::make_unique<dw::Shader>(GL_VERTEX_SHADER, g_sample_vs_src);
		m_fs = std::make_unique<dw::Shader>(GL_FRAGMENT_SHADER, g_sample_fs_src);

		if (!m_vs || !m_fs)
		{
			DW_LOG_FATAL("Failed to create Shaders");
			return false;
		}

		// Create general shader program
        dw::Shader* shaders[] = { m_vs.get(), m_fs.get() };
        m_program = std::make_unique<dw::Program>(2, shaders);

		if (!m_program)
		{
			DW_LOG_FATAL("Failed to create Shader Program");
			return false;
		}
        
        m_program->uniform_block_binding("ObjectUniforms", 1);
        
        // Create CSM shaders
        m_csm_vs = std::make_unique<dw::Shader>(GL_VERTEX_SHADER, g_csm_vs_src);
        m_csm_fs = std::make_unique<dw::Shader>(GL_FRAGMENT_SHADER, g_csm_fs_src);
        
        if (!m_csm_vs || !m_csm_fs)
        {
            DW_LOG_FATAL("Failed to create CSM Shaders");
            return false;
        }
        
        // Create CSM shader program
        dw::Shader* csm_shaders[] = { m_csm_vs.get(), m_csm_fs.get() };
        m_csm_program = std::make_unique<dw::Program>(2, csm_shaders);
        
        if (!m_csm_program)
        {
            DW_LOG_FATAL("Failed to create CSM Shader Program");
            return false;
        }
        
        m_csm_program->uniform_block_binding("ObjectUniforms", 1);

		// Create depth prepass shaders
		m_depth_prepass_vs = std::make_unique<dw::Shader>(GL_VERTEX_SHADER, g_depth_prepass_vs_src);
		m_depth_prepass_fs = std::make_unique<dw::Shader>(GL_FRAGMENT_SHADER, g_csm_fs_src);

		if (!m_depth_prepass_vs || !m_depth_prepass_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Prepass Shaders");
			return false;
		}

		// Create Depth prepass shader program
		dw::Shader* depth_prepass_shaders[] = { m_depth_prepass_vs.get(), m_depth_prepass_fs.get() };
		m_depth_prepass_program = std::make_unique<dw::Program>(2, depth_prepass_shaders);

		if (!m_depth_prepass_program)
		{
			DW_LOG_FATAL("Failed to create Depth Prepass Shader Program");
			return false;
		}

		m_depth_prepass_program->uniform_block_binding("ObjectUniforms", 1);

		// Create depth copy shaders
		m_depth_copy_vs = std::make_unique<dw::Shader>(GL_VERTEX_SHADER, g_post_process_vs_src);
		m_depth_copy_fs = std::make_unique<dw::Shader>(GL_FRAGMENT_SHADER, g_depth_copy_src);

		if (!m_depth_copy_vs || !m_depth_copy_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Copy Shaders");
			return false;
		}

		// Create Depth copy shader program
		dw::Shader* depth_copy_shaders[] = { m_depth_copy_vs.get(), m_depth_copy_fs.get() };
		m_depth_copy_program = std::make_unique<dw::Program>(2, depth_copy_shaders);

		if (!m_depth_copy_program)
		{
			DW_LOG_FATAL("Failed to create Depth Copy Shader Program");
			return false;
		}

		// Create depth reduction shaders
		m_depth_reduction_vs = std::make_unique<dw::Shader>(GL_VERTEX_SHADER, g_post_process_vs_src);
		m_depth_reduction_fs = std::make_unique<dw::Shader>(GL_FRAGMENT_SHADER, g_depth_reduction_fs_src);

		if (!m_depth_reduction_vs || !m_depth_reduction_fs)
		{
			DW_LOG_FATAL("Failed to create Depth Reduction Shaders");
			return false;
		}

		// Create Depth reduction shader program
		dw::Shader* depth_reduction_shaders[] = { m_depth_reduction_vs.get(), m_depth_reduction_fs.get() };
		m_depth_reduction_program = std::make_unique<dw::Program>(2, depth_reduction_shaders);

		if (!m_depth_reduction_program)
		{
			DW_LOG_FATAL("Failed to create Depth Reduction Shader Program");
			return false;
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_uniform_buffer()
	{
		// Create uniform buffer for object matrix data
        m_object_ubo = std::make_unique<dw::UniformBuffer>(GL_DYNAMIC_DRAW, sizeof(ObjectUniforms));
        
        // Create uniform buffer for global data
        m_global_ubo = std::make_unique<dw::ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(GlobalUniforms));
        
        // Create uniform buffer for CSM data
        m_csm_ubo = std::make_unique<dw::ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(CSMUniforms));

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

    void render_mesh(dw::Mesh* mesh, const ObjectUniforms& transforms, bool use_textures = true)
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
                submesh.mat->texture(0)->bind(0);

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
		// Update global uniforms.
		update_global_uniforms(m_global_uniforms);

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

			m_depth_reduction_fbos[i]->bind();
			glViewport(0, 0, m_width / scale, m_height / scale);
			
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
    std::unique_ptr<dw::Shader> m_vs;
	std::unique_ptr<dw::Shader> m_fs;
	std::unique_ptr<dw::Program> m_program;
	std::unique_ptr<dw::UniformBuffer> m_object_ubo;
    std::unique_ptr<dw::ShaderStorageBuffer> m_csm_ubo;
    std::unique_ptr<dw::ShaderStorageBuffer> m_global_ubo;
    
    // CSM shaders.
    std::unique_ptr<dw::Shader> m_csm_vs;
    std::unique_ptr<dw::Shader> m_csm_fs;
    std::unique_ptr<dw::Program> m_csm_program;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    std::unique_ptr<dw::Camera> m_debug_camera;

	// Depth Pre-pass
	std::unique_ptr<dw::Shader> m_depth_prepass_vs;
	std::unique_ptr<dw::Shader> m_depth_prepass_fs;
	std::unique_ptr<dw::Program> m_depth_prepass_program;
	std::unique_ptr<dw::Texture2D> m_depth_prepass_rt;
	std::unique_ptr<dw::Framebuffer> m_depth_prepass_fbo;

	std::unique_ptr<dw::Shader> m_depth_copy_vs;
	std::unique_ptr<dw::Shader> m_depth_copy_fs;
	std::unique_ptr<dw::Program> m_depth_copy_program;

	std::unique_ptr<dw::Shader> m_depth_reduction_vs;
	std::unique_ptr<dw::Shader> m_depth_reduction_fs;
	std::unique_ptr<dw::Program> m_depth_reduction_program;
	std::unique_ptr<dw::Texture2D> m_depth_reduction_rt;
	std::vector<std::unique_ptr<dw::Framebuffer>> m_depth_reduction_fbos;
    
	std::unique_ptr<dw::Shader> m_setup_shadows_cs;
	std::unique_ptr<dw::Program> m_setup_shadows_program;

	// Assets.
	dw::Mesh* m_plane;
    dw::Mesh* m_suzanne;

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
