#include <application.h>
#include <mesh.h>
#include <camera.h>
#include <material.h>
#include "csm.h"

// Embedded vertex shader source.
const char* g_sample_vs_src = R"(

layout (location = 0) in vec3 VS_IN_Position;
layout (location = 1) in vec2 VS_IN_TexCoord;
layout (location = 2) in vec3 VS_IN_Normal;
layout (location = 3) in vec3 VS_IN_Tangent;
layout (location = 4) in vec3 VS_IN_Bitangent;

layout (std140) uniform GlobalUniforms //#binding 0
{
    mat4 view;
    mat4 projection;
    mat4 crop;
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

layout (std140) uniform GlobalUniforms //#binding 0
{
    mat4 view;
    mat4 projection;
    mat4 crop;
};

layout (std140) uniform ObjectUniforms //#binding 1
{
    mat4 model;
};

void main()
{
    gl_Position = crop * model * vec4(VS_IN_Position, 1.0);
}

)";

// Embedded fragment shader source.
const char* g_sample_fs_src = R"(

out vec4 PS_OUT_Color;

in vec3 PS_IN_WorldFragPos;
in vec4 PS_IN_NDCFragPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

layout (std140) uniform CSMUniforms //#binding 2
{
    vec4 direction;
    int num_cascades;
    float far_bounds[8];
    mat4 texture_matrices[8];
};

uniform sampler2D s_Diffuse; //#slot 0
uniform sampler2DArray s_ShadowMap; //#slot 1

float shadow_occlussion(float frag_depth, vec3 n, vec3 l)
{
	int index = num_cascades - 1;

	// Find shadow cascade.
	for (int i = 0; i < num_cascades; i++)
	{
		if (frag_depth < far_bounds[i])
		{
			index = i;
			break;
		}
	}

	// Transform frag position into Light-space.
	vec4 light_space_pos = texture_matrices[index] * vec4(PS_IN_WorldFragPos, 1.0f);

	float shadow_map_depth = texture(s_ShadowMap, vec3(light_space_pos.xy, float(index))).r;
	float current_depth = light_space_pos.z;

	float bias = max(0.05 * (1.0 - dot(n, l)), 0.005);  
	
	float shadow = current_depth - bias > shadow_map_depth ? 1.0 : 0.0;

	return shadow;
}

vec3 debug_color(float frag_depth)
{
	int index = num_cascades - 1;

	// Find shadow cascade.
	for (int i = 0; i < num_cascades; i++)
	{
		if (frag_depth < far_bounds[i])
		{
			index = i;
			break;
		}
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

	vec3 diffuse = texture(s_Diffuse, PS_IN_TexCoord * 50).xyz;
	vec3 ambient = diffuse * 0.03;

	float frag_depth = (PS_IN_NDCFragPos.z / PS_IN_NDCFragPos.w) * 0.5 + 0.5;
	float shadow = shadow_occlussion(frag_depth, n, l);
	
	vec3 cascade = debug_color(frag_depth);
	vec3 color = (1.0 - shadow) * diffuse * lambert + ambient + cascade * 0.5;

    PS_OUT_Color = vec4(color, 1.0);
}

)";

const char* g_csm_fs_src = R"(

void main()
{
}

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
    DW_ALIGNED(16) glm::mat4 crop;
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
    DW_ALIGNED(16) int       num_cascades;
    DW_ALIGNED(16) FarBound  far_bounds[8];
    DW_ALIGNED(16) glm::mat4 texture_matrices[8];
};

class Sample : public dw::Application
{
protected:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	bool init(int argc, const char* argv[]) override
	{
		// Create GPU resources.
		if (!create_states())
			return false;

		if (!create_shaders())
			return false;

		if (!create_uniform_buffer())
			return false;

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
		// Update camera.
        update_camera();
        
        // Update CSM.
        m_csm.update(m_main_camera, m_csm_uniforms.direction);

		// Update transforms.
        update_transforms(m_debug_mode ? m_debug_camera : m_main_camera);

        // Render shadow map.
        render_shadow_map();
        
        // Render scene.
        render_scene();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void shutdown() override
	{
		// Cleanup CSM.
		m_csm.shutdown();

		// Destroy camera.
		DW_SAFE_DELETE(m_main_camera);
        DW_SAFE_DELETE(m_debug_camera);

		// Unload assets.
		dw::Mesh::unload(m_plane);
        dw::Mesh::unload(m_suzanne);

		// Cleanup GPU resources.
		m_device.destroy(m_object_ubo);
        m_device.destroy(m_global_ubo);
        m_device.destroy(m_csm_ubo);
        m_device.destroy(m_csm_program);
        m_device.destroy(m_csm_fs);
        m_device.destroy(m_csm_vs);
		m_device.destroy(m_program);
		m_device.destroy(m_fs);
		m_device.destroy(m_vs);
		m_device.destroy(m_ds);
		m_device.destroy(m_rs);
        m_device.destroy(m_shadow_map_rs);
		m_device.destroy(m_shadow_sampler);
		m_device.destroy(m_sampler);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void window_resized(int width, int height) override
	{
		// Override window resized method to update camera projection.
		m_main_camera->update_projection(60.0f, 0.1f, 100.0f, float(m_width) / float(m_height));

		// Re-initialize CSM to fit new frustum shape.
		initialize_csm();
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
        
        // Enable mouse look.
        if(code == GLFW_KEY_SPACE)
            m_mouse_look = true;
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
        
        // Disable mouse look.
        if(code == GLFW_KEY_SPACE)
            m_mouse_look = false;
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
	void initialize_csm()
	{
		m_csm_uniforms.direction = glm::vec4(glm::vec3(1.0f, -1.0f, 0.0f), 0.0f);
		m_csm_uniforms.direction = glm::normalize(m_csm_uniforms.direction);

		m_csm.initialize(&m_device, 0.75f, 100.0f, 4, 2048, m_main_camera, m_width, m_height, m_csm_uniforms.direction);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_shaders()
	{
		// Create general shaders
		m_vs = m_device.create_shader(g_sample_vs_src, ShaderType::VERTEX);
		m_fs = m_device.create_shader(g_sample_fs_src, ShaderType::FRAGMENT);

		if (!m_vs || !m_fs)
		{
			DW_LOG_FATAL("Failed to create Shaders");
			return false;
		}

		// Create general shader program
		Shader* shaders[] = { m_vs, m_fs };
		m_program = m_device.create_shader_program(shaders, 2);

		if (!m_program)
		{
			DW_LOG_FATAL("Failed to create Shader Program");
			return false;
		}
        
        // Create CSM shaders
        m_csm_vs = m_device.create_shader(g_csm_vs_src, ShaderType::VERTEX);
        m_csm_fs = m_device.create_shader(g_csm_fs_src, ShaderType::FRAGMENT);
        
        if (!m_csm_vs || !m_csm_fs)
        {
            DW_LOG_FATAL("Failed to create CSM Shaders");
            return false;
        }
        
        // Create CSM shader program
        Shader* csm_shaders[] = { m_csm_vs, m_csm_fs };
        m_csm_program = m_device.create_shader_program(csm_shaders, 2);
        
        if (!m_csm_program)
        {
            DW_LOG_FATAL("Failed to create CSM Shader Program");
            return false;
        }

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_states()
	{
		// Create rasterizer state
		RasterizerStateCreateDesc rs_desc;
		DW_ZERO_MEMORY(rs_desc);
		rs_desc.cull_mode = CullMode::BACK;
		rs_desc.fill_mode = FillMode::SOLID;
		rs_desc.front_winding_ccw = true;
		rs_desc.multisample = true;
		rs_desc.scissor = false;

		m_rs = m_device.create_rasterizer_state(rs_desc);
        
        // Create second rasterizer state with front-face culling for shadow mapping.
        rs_desc.cull_mode = CullMode::FRONT;
        
        m_shadow_map_rs = m_device.create_rasterizer_state(rs_desc);

		// Create depth stencil state
		DepthStencilStateCreateDesc ds_desc;
		DW_ZERO_MEMORY(ds_desc);
		ds_desc.depth_mask = true;
		ds_desc.enable_depth_test = true;
		ds_desc.enable_stencil_test = false;
		ds_desc.depth_cmp_func = ComparisonFunction::LESS_EQUAL;

		m_ds = m_device.create_depth_stencil_state(ds_desc);

		// Create sampler state.
		SamplerStateCreateDesc ss_desc;
		DW_ZERO_MEMORY(ss_desc);
		ss_desc.min_filter = TextureFilteringMode::ANISOTROPIC_ALL;
		ss_desc.mag_filter = TextureFilteringMode::ANISOTROPIC_ALL;
        ss_desc.max_anisotropy = 16;
		ss_desc.wrap_mode_u = TextureWrapMode::REPEAT;
		ss_desc.wrap_mode_v = TextureWrapMode::REPEAT;
		ss_desc.wrap_mode_w = TextureWrapMode::REPEAT;

		m_sampler = m_device.create_sampler_state(ss_desc);

		// Sampler state for Shadow map.
		ss_desc.min_filter = TextureFilteringMode::LINEAR;
		ss_desc.mag_filter = TextureFilteringMode::LINEAR;
		ss_desc.max_anisotropy = 0;
		ss_desc.wrap_mode_u = TextureWrapMode::CLAMP_TO_EDGE;
		ss_desc.wrap_mode_v = TextureWrapMode::CLAMP_TO_EDGE;
		ss_desc.wrap_mode_w = TextureWrapMode::CLAMP_TO_EDGE;

		m_shadow_sampler = m_device.create_sampler_state(ss_desc);

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool create_uniform_buffer()
	{
		// Create uniform buffer for object matrix data
		BufferCreateDesc uboDesc;
		DW_ZERO_MEMORY(uboDesc);
		uboDesc.data = nullptr;
		uboDesc.data_type = DataType::FLOAT;
		uboDesc.size = sizeof(ObjectUniforms);
		uboDesc.usage_type = BufferUsageType::DYNAMIC;

		m_object_ubo = m_device.create_uniform_buffer(uboDesc);
        
        // Create uniform buffer for global data
        uboDesc.size = sizeof(GlobalUniforms);
        
        m_global_ubo = m_device.create_uniform_buffer(uboDesc);
        
        // Create uniform buffer for CSM data
        uboDesc.size = sizeof(CSMUniforms);
        
        m_csm_ubo = m_device.create_uniform_buffer(uboDesc);

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool load_mesh()
	{
		m_plane = dw::Mesh::load("plane.obj", &m_device);
        m_suzanne = dw::Mesh::load("suzanne.obj", &m_device);
		return m_plane != nullptr && m_suzanne != nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void create_camera()
	{
		m_main_camera = new dw::Camera(60.0f, 0.1f, 100.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
        m_debug_camera = new dw::Camera(60.0f, 0.1f, 1000.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 5.0f, 20.0f), glm::vec3(0.0f, 0.0, -1.0f));
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

    void render_mesh(dw::Mesh* mesh, const ObjectUniforms& transforms, bool use_textures = true)
	{
        // Copy new data into UBO.
        update_object_uniforms(transforms);

		// Bind uniform buffers.
        m_device.bind_uniform_buffer(m_global_ubo, ShaderType::VERTEX, 0);
		m_device.bind_uniform_buffer(m_object_ubo, ShaderType::VERTEX, 1);

		// Bind vertex array.
		m_device.bind_vertex_array(mesh->mesh_vertex_array());

		// Set primitive type.
		m_device.set_primitive_type(PrimitiveType::TRIANGLES);

		// Bind sampler.
        if (use_textures)
            m_device.bind_sampler_state(m_sampler, ShaderType::FRAGMENT, 0);

		for (uint32_t i = 0; i < mesh->sub_mesh_count(); i++)
		{
			dw::SubMesh& submesh = mesh->sub_meshes()[i];

			// Bind texture.
            if (use_textures)
                m_device.bind_texture(submesh.mat->texture(0), ShaderType::FRAGMENT, 0);

			// Issue draw call.
			m_device.draw_indexed_base_vertex(submesh.index_count, submesh.base_index, submesh.base_vertex);
		}
	}
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void render_scene()
    {
        // Update global uniforms.
        update_global_uniforms(m_global_uniforms);
        
        // Update CSM uniforms.
        update_csm_uniforms(m_csm_uniforms);
        
        // Bind and clear framebuffer.
        m_device.bind_framebuffer(nullptr);
        m_device.set_viewport(m_width, m_height, 0, 0);
        
        // Clear default framebuffer.
        m_device.clear_framebuffer(ClearTarget::ALL, m_clear_color);
        
        // Bind states.
        m_device.bind_rasterizer_state(m_rs);
        m_device.bind_depth_stencil_state(m_ds);
        
        // Bind shader program.
        m_device.bind_shader_program(m_program);
        
        // Bind shadow map.
        m_device.bind_sampler_state(m_shadow_sampler, ShaderType::FRAGMENT, 1);
        m_device.bind_texture(m_csm.shadow_map(), ShaderType::FRAGMENT, 1);
        
        // Bind uniform buffers.
        m_device.bind_uniform_buffer(m_csm_ubo, ShaderType::FRAGMENT, 2);
        
        // Draw meshes.
        render_mesh(m_plane, m_plane_transforms);
        render_mesh(m_suzanne, m_suzanne_transforms);
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void render_shadow_map()
    {
        // Bind states.
        m_device.bind_depth_stencil_state(m_ds);
        
        // Bind shader program.
        m_device.bind_shader_program(m_csm_program);
        
        for (int i = 0; i < m_csm.frustum_split_count(); i++)
        {
            // Update global uniforms.
			m_global_uniforms.crop = m_csm.split_view_proj(i);
            
            update_global_uniforms(m_global_uniforms);
            
            // Bind and clear framebuffer.
            m_device.bind_framebuffer(m_csm.framebuffers()[i]);
            m_device.set_viewport(m_csm.shadow_map_size(), m_csm.shadow_map_size(), 0, 0);
            
            // Clear default framebuffer.
            m_device.clear_framebuffer(ClearTarget::ALL, m_clear_color);
            
            // Draw meshes. Disable textures because we don't need them here.
			m_device.bind_rasterizer_state(m_rs);
            render_mesh(m_plane, m_plane_transforms, false);

			m_device.bind_rasterizer_state(m_rs);
            render_mesh(m_suzanne, m_suzanne_transforms, false);
        }
    }

	// -----------------------------------------------------------------------------------------------------------------------------------

	void update_object_uniforms(const ObjectUniforms& transform)
	{
		void* ptr = m_device.map_buffer(m_object_ubo, BufferMapType::WRITE);
		memcpy(ptr, &transform, sizeof(ObjectUniforms));
		m_device.unmap_buffer(m_object_ubo);
	}
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_csm_uniforms(const CSMUniforms& csm)
    {
        void* ptr = m_device.map_buffer(m_csm_ubo, BufferMapType::WRITE);
        memcpy(ptr, &csm, sizeof(CSMUniforms));
        m_device.unmap_buffer(m_csm_ubo);
    }
    
    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_global_uniforms(const GlobalUniforms& global)
    {
        void* ptr = m_device.map_buffer(m_global_ubo, BufferMapType::WRITE);
        memcpy(ptr, &global, sizeof(GlobalUniforms));
        m_device.unmap_buffer(m_global_ubo);
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
        }
        
        // Update plane transforms.
        m_plane_transforms.model = glm::mat4(1.0f);

        // Update suzanne transforms.
        m_suzanne_transforms.model = glm::mat4(1.0f);
        m_suzanne_transforms.model = glm::translate(m_suzanne_transforms.model, glm::vec3(0.0f, 3.0f, 0.0f));
       // m_suzanne_transforms.model = glm::rotate(m_suzanne_transforms.model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
        m_suzanne_transforms.model = glm::scale(m_suzanne_transforms.model, glm::vec3(2.0f));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------
    
    void update_camera()
    {
        dw::Camera* current = m_main_camera;
        
        if (m_debug_mode)
            current = m_debug_camera;
        
        float forward_delta = m_heading_speed * m_delta;
        float right_delta = m_sideways_speed * m_delta;
        
        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);
        
        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_mouse_delta_y * m_camera_sensitivity * m_delta),
                                                    (float)(m_mouse_delta_x * m_camera_sensitivity * m_delta),
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

private:
	// Clear color.
	float m_clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

	// General GPU resources.
	Shader* m_vs;
	Shader* m_fs;
	ShaderProgram* m_program;
	UniformBuffer* m_object_ubo;
    UniformBuffer* m_csm_ubo;
    UniformBuffer* m_global_ubo;
    RasterizerState* m_rs;
	DepthStencilState* m_ds;
	SamplerState* m_sampler;
    
    // CSM shaders.
    Shader* m_csm_vs;
    Shader* m_csm_fs;
    ShaderProgram* m_csm_program;

    // Rasterizer state for CSM.
    RasterizerState* m_shadow_map_rs;
    
	// Shadow map sampler.
	SamplerState* m_shadow_sampler;

	// Assets.
	dw::Mesh* m_plane;
    dw::Mesh* m_suzanne;

	// Camera.
	dw::Camera* m_main_camera;
    dw::Camera* m_debug_camera;

	// Uniforms.
	ObjectUniforms m_plane_transforms;
    ObjectUniforms m_suzanne_transforms;
    GlobalUniforms m_global_uniforms;
    CSMUniforms    m_csm_uniforms;

	// Cascaded Shadow Mapping.
	CSM m_csm;
    
    // Camera controls.
    bool m_mouse_look = false;
    bool m_debug_mode = false;
    float m_heading_speed = 0.0f;
    float m_sideways_speed = 0.0f;
    float m_camera_sensitivity = 0.005f;
    float m_camera_speed = 0.01f;
};

DW_DECLARE_MAIN(Sample)
