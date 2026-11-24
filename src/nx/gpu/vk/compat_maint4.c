/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* vkGetDeviceImageMemoryRequirements and its sparse form without VK_KHR_maintenance4 (compat_maint4.h). The query of
   Vulkan 1.3 asks the device what an image of a create info would need, with no image. Here an image is made, which
   takes no memory until one is bound to it, asked, and destroyed. A create info that the device refuses gives
   requirements of size 0, and the caller's own vkCreateImage reports it. */
#include "nx/gpu/vk/compat_maint4.h"

static void compat_image_requirements(VkDevice dev, const VkDeviceImageMemoryRequirements* info,
                                      VkMemoryRequirements2* out) {
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(dev, info->pCreateInfo, NULL, &image) != VK_SUCCESS) {
        out->memoryRequirements = (VkMemoryRequirements){0};
        return;
    }
    /* a plane of a multi-planar image whose planes are bound apart */
    VkImagePlaneMemoryRequirementsInfo plane = {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
                                                .planeAspect = info->planeAspect};
    VkImageMemoryRequirementsInfo2 query = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                            .pNext = info->planeAspect ? &plane : NULL,
                                            .image = image};
    vkGetImageMemoryRequirements2(dev, &query, out);
    vkDestroyImage(dev, image, NULL);
}

static void compat_image_sparse_requirements(VkDevice dev, const VkDeviceImageMemoryRequirements* info,
                                             uint32_t* count, VkSparseImageMemoryRequirements2* out) {
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(dev, info->pCreateInfo, NULL, &image) != VK_SUCCESS) {
        *count = 0;
        return;
    }
    const VkImageSparseMemoryRequirementsInfo2 query = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2, .image = image};
    vkGetImageSparseMemoryRequirements2(dev, &query, count, out);
    vkDestroyImage(dev, image, NULL);
}

void vk_maint4_compat_install(void) {
    vkGetDeviceImageMemoryRequirements = compat_image_requirements;
    vkGetDeviceImageSparseMemoryRequirements = compat_image_sparse_requirements;
}
