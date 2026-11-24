/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* Dynamic rendering for a device that has Vulkan 1.2 and no VK_KHR_dynamic_rendering (compat_render.c): the calls of
   it that the renderer makes are made of render passes and framebuffers. */
#pragma once
#include "volk.h"

/* Point vkCmdBeginRendering, vkCmdEndRendering, vkCreateGraphicsPipelines, and the calls that make and destroy images
   and image views at the versions of compat_render.c. Call it after volk has loaded the device, and before the
   renderer makes its first image: the formats of the images and views are what a render pass is made from, and only
   the images and views made after this call are known. dev is the device the calls are for. The device must have
   vkCreateRenderPass2, which Vulkan 1.2 has in its core. Where it is called again, for another device, what it kept
   of the first one is dropped. */
void vk_render_compat_install(VkDevice dev);
