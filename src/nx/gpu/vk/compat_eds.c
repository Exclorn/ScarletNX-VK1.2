/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The calls of extended dynamic state for a device that has not the extensions (compat_eds.h). A draw sets cull mode,
   front face, topology, primitive restart, rasterizer discard, depth bias enable, the depth and stencil tests and
   their operations, and the count of viewports and scissors. Without the extensions a pipeline has them, in the
   PipeKey (vk.h PipeEds) that render.c fills, so the commands that set them stay as calls that do nothing: a pipeline
   would not list the state as dynamic, and the command would not be valid.

   Eden does the same (vk_graphics_pipeline.cpp: FixedPipelineState, and vk_state_tracker.cpp, which skips the
   commands where the device has not the extension). */
#include "nx/gpu/vk/compat_eds.h"

static void compat_cull_mode(VkCommandBuffer cb, VkCullModeFlags v) { (void)cb; (void)v; }
static void compat_front_face(VkCommandBuffer cb, VkFrontFace v) { (void)cb; (void)v; }
static void compat_topology(VkCommandBuffer cb, VkPrimitiveTopology v) { (void)cb; (void)v; }
static void compat_bool(VkCommandBuffer cb, VkBool32 v) { (void)cb; (void)v; }
static void compat_compare_op(VkCommandBuffer cb, VkCompareOp v) { (void)cb; (void)v; }
static void compat_stencil_op(VkCommandBuffer cb, VkStencilFaceFlags face, VkStencilOp fail, VkStencilOp pass,
                              VkStencilOp depth_fail, VkCompareOp compare) {
    (void)cb;
    (void)face;
    (void)fail;
    (void)pass;
    (void)depth_fail;
    (void)compare;
}
/* the pipeline holds the strides (has_dynamic_stride is off, so strides is NULL), and a buffer's size ends where the
   buffer does: a read past the range of a vertex binding is the buffer's, not zero */
static void compat_bind_vertex_buffers2(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* buffers,
                                        const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                                        const VkDeviceSize* strides) {
    (void)sizes;
    (void)strides;
    vkCmdBindVertexBuffers(cb, first, count, buffers, offsets);
}
static void compat_viewport_with_count(VkCommandBuffer cb, uint32_t count, const VkViewport* viewports) {
    vkCmdSetViewport(cb, 0, count, viewports);
}
static void compat_scissor_with_count(VkCommandBuffer cb, uint32_t count, const VkRect2D* scissors) {
    vkCmdSetScissor(cb, 0, count, scissors);
}

void vk_eds_compat_install(void) {
    vkCmdBindVertexBuffers2 = compat_bind_vertex_buffers2;
    vkCmdSetCullMode = compat_cull_mode;
    vkCmdSetFrontFace = compat_front_face;
    vkCmdSetPrimitiveTopology = compat_topology;
    vkCmdSetViewportWithCount = compat_viewport_with_count;
    vkCmdSetScissorWithCount = compat_scissor_with_count;
    vkCmdSetDepthTestEnable = compat_bool;
    vkCmdSetDepthWriteEnable = compat_bool;
    vkCmdSetDepthCompareOp = compat_compare_op;
    vkCmdSetDepthBoundsTestEnable = compat_bool;
    vkCmdSetStencilTestEnable = compat_bool;
    vkCmdSetStencilOp = compat_stencil_op;
    vkCmdSetDepthBiasEnable = compat_bool;
    vkCmdSetPrimitiveRestartEnable = compat_bool;
    vkCmdSetRasterizerDiscardEnable = compat_bool;
}
