/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The calls of VK_EXT_extended_dynamic_state and VK_EXT_extended_dynamic_state2 (Vulkan 1.3's core) that the renderer
   makes, for a device that has Vulkan 1.2 and not those extensions (compat_eds.c). The state they set is part of each
   pipeline there: PipeKey.eds holds it, pipeline.c puts it in the pipeline and takes it out of the dynamic states,
   as Eden does without the extension. */
#pragma once
#include "volk.h"

/* Point the calls at those that set nothing, except vkCmdBindVertexBuffers2 (vkCmdBindVertexBuffers, with the strides
   in the pipeline: g_vk.has_dynamic_stride is off), and vkCmdSetViewportWithCount and vkCmdSetScissorWithCount (the
   calls of Vulkan 1.0, for the count that the pipeline holds). Call it after volk has loaded the device, and only
   where g_vk.has_eds is off. */
void vk_eds_compat_install(void);
