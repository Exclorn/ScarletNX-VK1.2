/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The memory requirements of an image from its create info (VK_KHR_maintenance4, Vulkan 1.3's core), for a device that
   has Vulkan 1.2 and not that extension: compat_maint4.c makes an image of the create info, asks for its
   requirements, and destroys it. */
#pragma once
#include "volk.h"

/* Point vkGetDeviceImageMemoryRequirements and vkGetDeviceImageSparseMemoryRequirements at those. Call it after volk
   has loaded the device, and only where the device has no maintenance4. */
void vk_maint4_compat_install(void);
