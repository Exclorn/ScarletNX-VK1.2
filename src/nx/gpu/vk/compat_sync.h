/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The calls of VK_KHR_synchronization2 that the renderer makes, for a device that has Vulkan 1.2 and no such
   extension: each one is made of the calls that Vulkan 1.0 and 1.2 have (compat_sync.c). */
#pragma once

/* Point vkCmdPipelineBarrier2, vkQueueSubmit2 and vkCmdWriteTimestamp2 at those versions. Call it after volk has
   loaded the device, and only where the device has no synchronization2. The device must have timeline semaphores,
   which Vulkan 1.2 has in its core, as vkQueueSubmit2 passes their values through them. */
void vk_sync2_compat_install(void);
