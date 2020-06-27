/*
* Vulkan Example - Using descriptor sets for passing data to shader stages
*
* Relevant code parts are marked with [POI]
*
* Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glm/glm/gtx/intersect.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanModel.hpp"

#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	float shift = 0; // my code

	bool animate = false/*true*/;

	bool collision = false;

	bool pick = false;

	bool spheretest1 = false;
	bool spheretest2 = false;

	bool hitTriangle = false;
	bool OBBIntersection = false;

	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_NORMAL,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
	});

	struct Cube
	{
		struct Matrices 
		{
			glm::mat4 projection;
			glm::mat4 view;
			glm::mat4 model;
		} matrices;
		VkDescriptorSet descriptorSet;
		vks::Texture2D texture;
		vks::Buffer uniformBuffer;
		glm::vec3 rotation;
	};
	std::array<Cube, 2> cubes;

	struct Models
	{
		vks::Model cube;
	} models;
	
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;

	VkDescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Using descriptor Sets";
		settings.overlay = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -5.0f));
	}

	~VulkanExample()
	{
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		models.cube.destroy();
		for (auto cube : cubes) 
		{
			cube.uniformBuffer.destroy();
			cube.texture.destroy();
		}
	}

	virtual void getEnabledFeatures()
	{
		if (deviceFeatures.samplerAnisotropy) 
		{
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		};
	}
	VkViewport viewportglob;
	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) 
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			viewportglob = viewport;
			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &models.cube.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(drawCmdBuffers[i], models.cube.indices.buffer, 0, VK_INDEX_TYPE_UINT32);

			/* 
				[POI] Render cubes with separate descriptor sets
			*/
			for (auto cube : cubes) 
			{
				// Bind the cube's descriptor set. This tells the command buffer to use the uniform buffer and image set for this cube
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &cube.descriptorSet, 0, nullptr);
				vkCmdDrawIndexed(drawCmdBuffers[i], models.cube.indexCount, 1, 0, 0, 0);
			}

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadAssets()
	{
		models.cube.loadFromFile(getAssetPath() + "models/cube.dae", vertexLayout, 1.0f, vulkanDevice, queue);
		cubes[0].texture.loadFromFile(getAssetPath() + "textures/crate01_color_height_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
		cubes[1].texture.loadFromFile(getAssetPath() + "textures/crate02_color_height_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
	}

	/*
		[POI] Set up descriptor sets and set layout
	*/
	void setupDescriptors()
	{
		/*

			Descriptor set layout
			
			The layout describes the shader bindings and types used for a certain descriptor layout and as such must match the shader bindings

			Shader bindings used in this example:

			VS:
				layout (set = 0, binding = 0) uniform UBOMatrices ...

			FS :
				layout (set = 0, binding = 1) uniform sampler2D ...;

		*/

		std::array<VkDescriptorSetLayoutBinding,2> setLayoutBindings{};

		/*
			Binding 0: Uniform buffers (used to pass matrices matrices)
		*/
		setLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// Shader binding point
		setLayoutBindings[0].binding = 0;
		// Accessible from the vertex shader only (flags can be combined to make it accessible to multiple shader stages)
		setLayoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		// Binding contains one element (can be used for array bindings)
		setLayoutBindings[0].descriptorCount = 1;

		/*
			Binding 1: Combined image sampler (used to pass per object texture information) 
		*/
		setLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		setLayoutBindings[1].binding = 1;
		// Accessible from the fragment shader only
		setLayoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		setLayoutBindings[1].descriptorCount = 1;

		// Create the descriptor set layout
		VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{};
		descriptorLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
		descriptorLayoutCI.pBindings = setLayoutBindings.data();
		
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayoutCI, nullptr, &descriptorSetLayout));

		/*

			Descriptor pool

			Actual descriptors are allocated from a descriptor pool telling the driver what types and how many
			descriptors this application will use

			An application can have multiple pools (e.g. for multiple threads) with any number of descriptor types
			as long as device limits are not surpassed

			It's good practice to allocate pools with actually required descriptor types and counts

		*/

		std::array<VkDescriptorPoolSize, 2> descriptorPoolSizes{};

		// Uniform buffers : 1 for scene and 1 per object (scene and local matrices)
		descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorPoolSizes[0].descriptorCount = 1 + static_cast<uint32_t>(cubes.size());

		// Combined image samples : 1 per mesh texture
		descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorPoolSizes[1].descriptorCount = static_cast<uint32_t>(cubes.size());

		// Create the global descriptor pool
		VkDescriptorPoolCreateInfo descriptorPoolCI = {};
		descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
		descriptorPoolCI.pPoolSizes = descriptorPoolSizes.data();
		// Max. number of descriptor sets that can be allocted from this pool (one per object)
		descriptorPoolCI.maxSets = static_cast<uint32_t>(descriptorPoolSizes.size());
		
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorPool));

		/*

			Descriptor sets

			Using the shared descriptor set layout and the descriptor pool we will now allocate the descriptor sets.

			Descriptor sets contain the actual descriptor fo the objects (buffers, images) used at render time.

		*/

		for (auto &cube: cubes) 
		{

			// Allocates an empty descriptor set without actual descriptors from the pool using the set layout
			VkDescriptorSetAllocateInfo allocateInfo{};
			allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocateInfo.descriptorPool = descriptorPool;
			allocateInfo.descriptorSetCount = 1;
			allocateInfo.pSetLayouts = &descriptorSetLayout;
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocateInfo, &cube.descriptorSet));

			// Update the descriptor set with the actual descriptors matching shader bindings set in the layout

			std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};

			/*
				Binding 0: Object matrices Uniform buffer
			*/
			writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSets[0].dstSet = cube.descriptorSet;
			writeDescriptorSets[0].dstBinding = 0;
			writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			writeDescriptorSets[0].pBufferInfo = &cube.uniformBuffer.descriptor;
			writeDescriptorSets[0].descriptorCount = 1;

			/*
				Binding 1: Object texture
			*/
			writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSets[1].dstSet = cube.descriptorSet;
			writeDescriptorSets[1].dstBinding = 1;
			writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			// Images use a different descriptor strucutre, so we use pImageInfo instead of pBufferInfo
			writeDescriptorSets[1].pImageInfo = &cube.texture.descriptor;
			writeDescriptorSets[1].descriptorCount = 1;

			// Execute the writes to update descriptors for this set
			// Note that it's also possible to gather all writes and only run updates once, even for multiple sets
			// This is possible because each VkWriteDescriptorSet also contains the destination set to be updated
			// For simplicity we will update once per set instead

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}

	}

	void preparePipelines()
	{
		/*
			[POI] Create a pipeline layout used for our graphics pipeline 
		*/
		VkPipelineLayoutCreateInfo pipelineLayoutCI{};
		pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		// The pipeline layout is based on the descriptor set layout we created above
		pipelineLayoutCI.setLayoutCount = 1;
		pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);

		// Vertex bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = 
		{
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = 
		{
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 0: Position			
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6),		// Location 2: UV
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8),	// Location 3: Color
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		VkGraphicsPipelineCreateInfo pipelineCreateInfoCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCreateInfoCI.pVertexInputState = &vertexInputState;
		pipelineCreateInfoCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCreateInfoCI.pRasterizationState = &rasterizationStateCI;
		pipelineCreateInfoCI.pColorBlendState = &colorBlendStateCI;
		pipelineCreateInfoCI.pMultisampleState = &multisampleStateCI;
		pipelineCreateInfoCI.pViewportState = &viewportStateCI;
		pipelineCreateInfoCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCreateInfoCI.pDynamicState = &dynamicStateCI;

		const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages =
		{
			loadShader(getAssetPath() + "shaders/descriptorsets/cube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getAssetPath() + "shaders/descriptorsets/cube.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		pipelineCreateInfoCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfoCI.pStages = shaderStages.data();

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfoCI, nullptr, &pipeline));
	}

	void prepareUniformBuffers()
	{
		// Vertex shader matrix uniform buffer block
		for (auto& cube : cubes)
		{
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
																&cube.uniformBuffer, sizeof(Cube::Matrices)));
			VK_CHECK_RESULT(cube.uniformBuffer.map());
		}

		updateUniformBuffers();
	}
	glm::vec3 pickingPos;
	glm::vec3 pickingDir;
	glm::vec3 pickingDirNormalize;
	glm::mat4 cube2Modelmatrix;
	void updateUniformBuffers()
	{
		cubes[0].matrices.model = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f/*0.0f*/ /*+ shift*/ /*my code*/, 0.0f, 0.0f));
		cubes[1].matrices.model = glm::translate(glm::mat4(1.0f), glm::vec3( 1.5f + shift, 0.5f, 0.0f));		

		for (auto& cube : cubes)
		{
			cube.matrices.projection = camera.matrices.perspective;
			cube.matrices.view = camera.matrices.view;

			auto cubed = cube.matrices.projection * cube.matrices.view * cube.matrices.model * glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

			cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
			cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
			cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

			cube2Modelmatrix = cube.matrices.model;		// this will be cubes[1].matrices.model, because is second(last) in iteration
			cube2Modelmatrix[3][1] = cube2Modelmatrix[3][1] * -1;
			memcpy(cube.uniformBuffer.mapped, &cube.matrices, sizeof(cube.matrices));
		}				

		// https://stackoverflow.com/questions/35261192/how-to-use-glmproject-to-get-the-coordinates-of-a-point-in-world-space
		glm::vec4 viewportloc = glm::vec4(0/*viewportglob.x*/, 0/*viewportglob.y*/, viewportglob.width, viewportglob.height);

		glm::vec3 projected = glm::project(glm::vec3(1.5f + shift, 0.5f, 0.0f), camera.matrices.view/*cubes[1].matrices.model*/, cubes[1].matrices.projection, viewportloc);
		glm::vec3 unprojected = glm::unProject(projected, camera.matrices.view/*cubes[1].matrices.model*/, cubes[1].matrices.projection, viewportloc);
		
				
		glm::vec3 projected2 = glm::project(glm::vec3(-2.0f, 0.00000001f, 0.0f), camera.matrices.view/*cubes[0].matrices.model*/, cubes[0].matrices.projection, viewportloc);
		glm::vec3 unprojected2 = glm::unProject(projected2, camera.matrices.view/*cubes[0].matrices.model*/, cubes[0].matrices.projection, viewportloc);

		
		if ((unprojected.x - 1 <= unprojected2.x + 1) && (unprojected.x - 1 >= unprojected2.x - 1) ||
			(unprojected.x + 1 >= unprojected2.x - 1) && (unprojected.x + 1 <= unprojected2.x + 1))
		{
			collision = true;
		}


		// https://stackoverflow.com/questions/30340558/how-to-get-object-coordinates-from-screen-coordinates
		glm::vec2 screenPos(mousePos.x, mousePos.y);
		screenPos.y = height - screenPos.y;				
	

		glm::vec3 a(screenPos.x, screenPos.y, 0);
		glm::vec3 b(screenPos.x, screenPos.y, 1);	

				
		//glm::vec3 cameraPos = glm::vec3();
		//glm::mat4 view = glm::lookAt(glm::vec3(0, 0, -5.0f/*-zoom*/), cameraPos, glm::vec3(0, 1, 0));		
		

		//glm::vec3 result = glm::unProject(a, view, proj, viewportloc);
		//glm::vec3 result2 = glm::unProject(b, view, proj, viewportloc);	

		glm::vec3 result = glm::unProject(a, camera.matrices.view, camera.matrices.perspective, viewportloc);
		glm::vec3 result2 = glm::unProject(b, camera.matrices.view, camera.matrices.perspective, viewportloc);
		
		

		

		/*glm::vec3*/ pickingPos = result;
		/*glm::vec3*/ pickingDir = result2 - result;
		/*glm::vec3*/ pickingDirNormalize = glm::normalize(pickingDir);		

		models.cube.vertexCount;
		models.cube.vertices;
		models.cube.parts[0].vertexBase;

		vertexLayout.components[2];

		models.cube.indices;
		//-----------------------------------------------------------------------------
		// https://stackoverflow.com/questions/12678225/intersection-problems-with-ray-sphere-intersection
		//Create sphere with center in (0, 0, -20) and with radius 10
		Sphere testSphere(glm::vec3(-2.0f, 0.0f, 0.0f), 1.0f);
	RayR testRay(pickingPos, pickingDirNormalize);
		if (testSphere.intersection(testRay))
		{
			//Increase counter
			spheretest1 = true;
		}	

		glm::vec2 bary;
		float Distance;
		float vcv = vks::vertexBufferPositions[0];
		for (int j = 0; j < vks::vertexBufferPositions.size(); j = j + 9)
		{
			glm::vec3 v0 = glm::vec3(vks::vertexBufferPositions[j + 0], vks::vertexBufferPositions[j + 1], vks::vertexBufferPositions[j + 2]);
			glm::vec3 v1 = glm::vec3(vks::vertexBufferPositions[j + 3], vks::vertexBufferPositions[j + 4], vks::vertexBufferPositions[j + 5]);
			glm::vec3 v2 = glm::vec3(vks::vertexBufferPositions[j + 6], vks::vertexBufferPositions[j + 7], vks::vertexBufferPositions[j + 8]);

			v0 = v0 + glm::vec3(1.5f + shift, -0.5f, 0.0f);
			v1 = v1 + glm::vec3(1.5f + shift, -0.5f, 0.0f);
			v2 = v2 + glm::vec3(1.5f + shift, -0.5f, 0.0f);

			

			if (glm::intersectRayTriangle(pickingPos, pickingDirNormalize, v0, v1, v2, bary, Distance))
			{
				hitTriangle = true;
			}

		}
		for (auto& component : vks::vertexBufferPositions)
		{
			float s = component;
		}
			//glm::intersectRayTriangle(pickingPos, pickingDirNormalize, component, component, component, bary);

		//-----------------------------------------------------------------------------		
		float intersectionDistance;

		if (glm::intersectRaySphere(pickingPos, pickingDirNormalize, glm::vec3(1.5f, 0.5f, 0.0f), 1, intersectionDistance))
		{
			float z = intersectionDistance;

			spheretest2 = true;
		}

		//-----------------------------------------------------------------------------		
		// http://www.scratchapixel.com/code.php?id=10&origin=/lessons/3d-basic-rendering/ray-tracing-rendering-simple-shapes&src=1
		AABBox box(glm::vec3(-1), glm::vec3(1));

		// glm::scale

		Ray ray(pickingPos, pickingDirNormalize);
		float t;
		if (box.intersect(ray, t))
		{
			glm::vec3 Phit = ray.orig + ray.dir * t;			
			int t = 0;

			pick = true;
		}


		float intersection_distance; // Output of TestRayOBBIntersection()
		glm::vec3 aabb_min(-1.0f, -1.0f, -1.0f);
		glm::vec3 aabb_max(1.0f, 1.0f, 1.0f);

		//http://www.opengl-tutorial.org/miscellaneous/clicking-on-objects/picking-with-custom-ray-obb-function/
		if (TestRayOBBIntersection(pickingPos, pickingDirNormalize, aabb_min, aabb_max, cube2Modelmatrix, intersection_distance))
		{			
			OBBIntersection = true;
		}
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (animate)
		{
			cubes[0].rotation.x += 2.5f * frameTimer;
			if (cubes[0].rotation.x > 360.0f)
				cubes[0].rotation.x -= 360.0f;
			cubes[1].rotation.y += 2.0f * frameTimer;
			if (cubes[1].rotation.x > 360.0f)
				cubes[1].rotation.x -= 360.0f;
		}
		//if ((camera.updated) || (animate)) 
		{
			updateUniformBuffers();
		}
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings"))
		{
			overlay->checkBox("Animate", &animate);
		}

		if (collision)
		{
			overlay->text("Collision");
			collision = false;
		}
		else 
			overlay->text("No collision");



		std::ostringstream oss;
		oss << "pickingPos: x= " << pickingPos.x << " y= " << pickingPos.y << " z= " << pickingPos.z;
		std::string var = oss.str();
		const char* str = var.c_str();
		overlay->text(str);

		std::ostringstream oss2;
		oss2 << "pickingDir: x= " << pickingDir.x << " y= " << pickingDir.y << " z= " << pickingDir.z;
		var = oss2.str();
		str = var.c_str();
		overlay->text(str);
		
		std::ostringstream oss3;
		oss3 << "pickingDirNormalize: x= " << pickingDirNormalize.x << " y= " << pickingDirNormalize.y << " z= " << pickingDirNormalize.z;
		var = oss3.str();
		str = var.c_str();
		overlay->text(str);
				


		if (pick)
		{
			overlay->text("Cube pick");
			pick = false;
		}
		else
			overlay->text("Cube no pick");
		
		if (spheretest1)
		{
			overlay->text("spheretest1 intersect");
			spheretest1 = false;
		}
		else
			overlay->text("spheretest1 no intersect");

		if (spheretest2)
		{
			overlay->text("spheretest2 intersect");
			spheretest2 = false;
		}
		else
			overlay->text("spheretest2 no intersect");


		if (hitTriangle)
		{
			overlay->text("Triangle hit");
			hitTriangle = false;
		}
		else
			overlay->text("Triangle no hit");

		if (OBBIntersection)
		{
			overlay->text("OBBIntersection true");
			OBBIntersection = false;
		}
		else
			overlay->text("OBBIntersection false");


		
	}

	// my code
	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_B:		
			shift -= 0.5f;
			break;
		case KEY_N:		
			shift += 0.5f;
			break;		
		}
	}

	class Ray
	{
	public:
		Ray(const glm::vec3 &orig, const glm::vec3 &dir) : orig(orig), dir(dir)
		{
			invdir = glm::vec3(1) / dir;
			sign[0] = (invdir.x < 0);
			sign[1] = (invdir.y < 0);
			sign[2] = (invdir.z < 0);
		}
		glm::vec3 orig, dir; // ray orig and dir
		glm::vec3 invdir;
		int sign[3];
	};

	class AABBox
	{
	public:
		AABBox(const glm::vec3 &b0, const glm::vec3 &b1)
		{ 
			bounds[0] = b0,
			bounds[1] = b1;
		}

		bool intersect(const Ray &r, float &t) const
		{
			float tmin, tmax, tymin, tymax, tzmin, tzmax;

			tmin = (bounds[r.sign[0]].x - r.orig.x) * r.invdir.x;
			tmax = (bounds[1 - r.sign[0]].x - r.orig.x) * r.invdir.x;
			tymin = (bounds[r.sign[1]].y - r.orig.y) * r.invdir.y;
			tymax = (bounds[1 - r.sign[1]].y - r.orig.y) * r.invdir.y;

			if ((tmin > tymax) || (tymin > tmax))
				return false;

			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			tzmin = (bounds[r.sign[2]].z - r.orig.z) * r.invdir.z;
			tzmax = (bounds[1 - r.sign[2]].z - r.orig.z) * r.invdir.z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return false;

			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;

			t = tmin;

			if (t < 0) 
			{
				t = tmax;
				if (t < 0) return false;
			}

			return true;
		}
		glm::vec3 bounds[2];
	};

	class RayR
	{
	public:
		glm::vec3 m_origin;
		glm::vec3 m_direction;

		RayR::RayR(glm::vec3 origin, glm::vec3 direction)
			: m_origin(origin), m_direction(direction)
		{
		}

		glm::vec3 origin()
		{
			return m_origin;
		}

		glm::vec3 direction()
		{
			return m_direction;
		}
	};

	class Sphere
	{
	public:
		glm::vec3 m_center;
		float m_radius, m_radiusSquared;

		Sphere::Sphere(glm::vec3 center, float radius) : m_center(center), m_radius(radius), m_radiusSquared(radius*radius)
		{
		}

		//Sphere-ray intersection. Equation: (P-C)^2 - R^2 = 0, P = o+t*d
		//(P-C)^2 - R^2 => (o+t*d-C)^2-R^2 => o^2+(td)^2+C^2+2td(o-C)-2oC-R^2
		//=> at^2+bt+c, a = d*d, b = 2d(o-C), c = (o-C)^2-R^2
		//o = ray origin, d = ray direction, C = sphere center, R = sphere radius
		bool Sphere::intersection(RayR& ray) const
		{
			//Squared distance between ray origin and sphere center
			float squaredDist = glm::dot(ray.origin() - m_center, ray.origin() - m_center);

			//If the distance is less than the squared radius of the sphere...
			if (squaredDist <= m_radiusSquared)
			{
				//Point is in sphere, consider as no intersection existing
				//std::cout << "Point inside sphere..." << std::endl;
				return false;
			}

			//Will hold solution to quadratic equation
			float t0, t1;

			//Calculating the coefficients of the quadratic equation
			float a = glm::dot(ray.direction(), ray.direction()); // a = d*d
			float b = 2.0f*glm::dot(ray.direction(), ray.origin() - m_center); // b = 2d(o-C)
			float c = glm::dot(ray.origin() - m_center, ray.origin() - m_center) - m_radiusSquared; // c = (o-C)^2-R^2

																									//Calculate discriminant
			float disc = (b*b) - (4.0f*a*c);

			if (disc < 0) //If discriminant is negative no intersection happens
			{
				//std::cout << "No intersection with sphere..." << std::endl;
				return false;
			}
			else //If discriminant is positive one or two intersections (two solutions) exists
			{
				float sqrt_disc = glm::sqrt(disc);
				t0 = (-b - sqrt_disc) / (2 * a);
				t1 = (-b + sqrt_disc) / (2 * a);
			}

			//If the second intersection has a negative value then the intersections
			//happen behind the ray origin which is not considered. Otherwise t0 is
			//the intersection to be considered
			if (t1<0)
			{
				//std::cout << "No intersection with sphere..." << std::endl;
				return false;
			}
			else
			{
				//std::cout << "Intersection with sphere..." << std::endl;
				return true;
			}
		}
	};


	bool TestRayOBBIntersection(
		glm::vec3 ray_origin,        // Ray origin, in world space
		glm::vec3 ray_direction,     // Ray direction (NOT target position!), in world space. Must be normalize()'d.
		glm::vec3 aabb_min,          // Minimum X,Y,Z coords of the mesh when not transformed at all.
		glm::vec3 aabb_max,          // Maximum X,Y,Z coords. Often aabb_min*-1 if your mesh is centered, but it's not always the case.
		glm::mat4 ModelMatrix,       // Transformation applied to the mesh (which will thus be also applied to its bounding box)
		float& intersection_distance // Output : distance between ray_origin and the intersection with the OBB
	) 
	{

		// Intersection method from Real-Time Rendering and Essential Mathematics for Games

		float tMin = 0.0f;
		float tMax = 100000.0f;

		glm::vec3 OBBposition_worldspace(ModelMatrix[3].x, ModelMatrix[3].y, ModelMatrix[3].z);

		glm::vec3 delta = OBBposition_worldspace - ray_origin;

		// Test intersection with the 2 planes perpendicular to the OBB's X axis
		{
			glm::vec3 xaxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
			float e = glm::dot(xaxis, delta);
			float f = glm::dot(ray_direction, xaxis);

			if (fabs(f) > 0.001f) { // Standard case

				float t1 = (e + aabb_min.x) / f; // Intersection with the "left" plane
				float t2 = (e + aabb_max.x) / f; // Intersection with the "right" plane
												 // t1 and t2 now contain distances betwen ray origin and ray-plane intersections

												 // We want t1 to represent the nearest intersection, 
												 // so if it's not the case, invert t1 and t2
				if (t1>t2) 
				{
					float w = t1; t1 = t2; t2 = w; // swap t1 and t2
				}

				// tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
				if (t2 < tMax)
					tMax = t2;
				// tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
				if (t1 > tMin)
					tMin = t1;

				// And here's the trick :
				// If "far" is closer than "near", then there is NO intersection.
				// See the images in the tutorials for the visual explanation.
				if (tMax < tMin)
					return false;

			}
			else
			{ // Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
				if (-e + aabb_min.x > 0.0f || -e + aabb_max.x < 0.0f)
					return false;
			}
		}


		// Test intersection with the 2 planes perpendicular to the OBB's Y axis
		// Exactly the same thing than above.
		{
			glm::vec3 yaxis(ModelMatrix[1].x, ModelMatrix[1].y, ModelMatrix[1].z);
			float e = glm::dot(yaxis, delta);
			float f = glm::dot(ray_direction, yaxis);

			if (fabs(f) > 0.001f)
			{

				float t1 = (e + aabb_min.y) / f;
				float t2 = (e + aabb_max.y) / f;

				if (t1>t2) { float w = t1; t1 = t2; t2 = w; }

				if (t2 < tMax)
					tMax = t2;
				if (t1 > tMin)
					tMin = t1;
				if (tMin > tMax)
					return false;

			}
			else 
			{
				if (-e + aabb_min.y > 0.0f || -e + aabb_max.y < 0.0f)
					return false;
			}
		}


		// Test intersection with the 2 planes perpendicular to the OBB's Z axis
		// Exactly the same thing than above.
		{
			glm::vec3 zaxis(ModelMatrix[2].x, ModelMatrix[2].y, ModelMatrix[2].z);
			float e = glm::dot(zaxis, delta);
			float f = glm::dot(ray_direction, zaxis);

			if (fabs(f) > 0.001f)
			{

				float t1 = (e + aabb_min.z) / f;
				float t2 = (e + aabb_max.z) / f;

				if (t1>t2) { float w = t1; t1 = t2; t2 = w; }

				if (t2 < tMax)
					tMax = t2;
				if (t1 > tMin)
					tMin = t1;
				if (tMin > tMax)
					return false;

			}
			else
			{
				if (-e + aabb_min.z > 0.0f || -e + aabb_max.z < 0.0f)
					return false;
			}
		}

		intersection_distance = tMin;
		return true;
	}

	
};



VULKAN_EXAMPLE_MAIN()