/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* synchronization2 for a device without it (compat_sync.h). What changes from the 1.0 calls:
   - A barrier of Vulkan 1.0 has one pair of stage masks for the whole call, and synchronization2 has one pair for
     each barrier. The call here uses the union of all pairs. That waits for more than a barrier needs, never for less.
   - The stage and access masks have 64 bits. The low 32 bits mean the same in both. The stages and accesses that
     synchronization2 added (copy, blit, shader sampled read, and so on) are folded into the 1.0 stage or access
     that contained them, and a stage with no 1.0 name becomes ALL_COMMANDS.
   - A submit passes the values of its timeline semaphores in VkTimelineSemaphoreSubmitInfo. */
#include "nx/gpu/vk/compat_sync.h"
#include "common/utilities/bits.h"
#include "volk.h"
#include <stdint.h>
#include <stdlib.h>

#define LOW_BITS 0xffffffffull

/* the stages that synchronization2 added and the 1.0 ones that they fold into */
#define COPY_STAGES                                                                                                    \
    (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT |                  \
     VK_PIPELINE_STAGE_2_CLEAR_BIT)
#define INPUT_STAGES (VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT)
#define PRE_RASTER_STAGES VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT

static VkPipelineStageFlags stage_v1(VkPipelineStageFlags2 s) {
    VkPipelineStageFlags r = (VkPipelineStageFlags)(s & LOW_BITS);
    if (s & COPY_STAGES) r |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (s & INPUT_STAGES) r |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    if (s & PRE_RASTER_STAGES)
        r |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
             VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    /* a stage that Vulkan 1.0 has no name for: all of them */
    if (s & ~(LOW_BITS | COPY_STAGES | INPUT_STAGES | PRE_RASTER_STAGES)) r |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    return r;
}

static VkAccessFlags access_v1(VkAccessFlags2 a) {
    VkAccessFlags r = (VkAccessFlags)(a & LOW_BITS);
    if (a & (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT)) r |= VK_ACCESS_SHADER_READ_BIT;
    if (a & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) r |= VK_ACCESS_SHADER_WRITE_BIT;
    return r;
}

enum { BATCH = 16 };

static void VKAPI_CALL compat_cmd_pipeline_barrier2(VkCommandBuffer cb, const VkDependencyInfo* d) {
    VkPipelineStageFlags src = 0, dst = 0;
    for (uint32_t i = 0; i < d->memoryBarrierCount; i++) {
        src |= stage_v1(d->pMemoryBarriers[i].srcStageMask);
        dst |= stage_v1(d->pMemoryBarriers[i].dstStageMask);
    }
    for (uint32_t i = 0; i < d->bufferMemoryBarrierCount; i++) {
        src |= stage_v1(d->pBufferMemoryBarriers[i].srcStageMask);
        dst |= stage_v1(d->pBufferMemoryBarriers[i].dstStageMask);
    }
    for (uint32_t i = 0; i < d->imageMemoryBarrierCount; i++) {
        src |= stage_v1(d->pImageMemoryBarriers[i].srcStageMask);
        dst |= stage_v1(d->pImageMemoryBarriers[i].dstStageMask);
    }
    /* no stage on one side: nothing to wait for there, which these two stages say */
    if (!src) src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (!dst) dst = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    /* BATCH barriers of each kind for a call, and a call again for the rest: the barriers do not depend on each
       other, so the split changes nothing */
    uint32_t mi = 0, bi = 0, ii = 0;
    do {
        VkMemoryBarrier m[BATCH];
        VkBufferMemoryBarrier b[BATCH];
        VkImageMemoryBarrier im[BATCH];
        const uint32_t mc = snx_min_u32(BATCH, d->memoryBarrierCount - mi);
        const uint32_t bc = snx_min_u32(BATCH, d->bufferMemoryBarrierCount - bi);
        const uint32_t ic = snx_min_u32(BATCH, d->imageMemoryBarrierCount - ii);
        for (uint32_t i = 0; i < mc; i++) {
            const VkMemoryBarrier2* s = &d->pMemoryBarriers[mi + i];
            m[i] = (VkMemoryBarrier){.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                     .pNext = s->pNext,
                                     .srcAccessMask = access_v1(s->srcAccessMask),
                                     .dstAccessMask = access_v1(s->dstAccessMask)};
        }
        for (uint32_t i = 0; i < bc; i++) {
            const VkBufferMemoryBarrier2* s = &d->pBufferMemoryBarriers[bi + i];
            b[i] = (VkBufferMemoryBarrier){.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                           .pNext = s->pNext,
                                           .srcAccessMask = access_v1(s->srcAccessMask),
                                           .dstAccessMask = access_v1(s->dstAccessMask),
                                           .srcQueueFamilyIndex = s->srcQueueFamilyIndex,
                                           .dstQueueFamilyIndex = s->dstQueueFamilyIndex,
                                           .buffer = s->buffer,
                                           .offset = s->offset,
                                           .size = s->size};
        }
        for (uint32_t i = 0; i < ic; i++) {
            const VkImageMemoryBarrier2* s = &d->pImageMemoryBarriers[ii + i];
            im[i] = (VkImageMemoryBarrier){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                           .pNext = s->pNext,
                                           .srcAccessMask = access_v1(s->srcAccessMask),
                                           .dstAccessMask = access_v1(s->dstAccessMask),
                                           .oldLayout = s->oldLayout,
                                           .newLayout = s->newLayout,
                                           .srcQueueFamilyIndex = s->srcQueueFamilyIndex,
                                           .dstQueueFamilyIndex = s->dstQueueFamilyIndex,
                                           .image = s->image,
                                           .subresourceRange = s->subresourceRange};
        }
        vkCmdPipelineBarrier(cb, src, dst, d->dependencyFlags, mc, m, bc, b, ic, im);
        mi += mc;
        bi += bc;
        ii += ic;
    } while (mi < d->memoryBarrierCount || bi < d->bufferMemoryBarrierCount || ii < d->imageMemoryBarrierCount);
}

static VkResult VKAPI_CALL compat_queue_submit2(VkQueue queue, uint32_t count, const VkSubmitInfo2* in, VkFence fence) {
    size_t waits = 0, signals = 0, buffers = 0;
    for (uint32_t i = 0; i < count; i++) {
        waits += in[i].waitSemaphoreInfoCount;
        signals += in[i].signalSemaphoreInfoCount;
        buffers += in[i].commandBufferInfoCount;
    }
    /* one block for the arrays that the submits point into: +1 so that no size is 0 */
    VkSubmitInfo* si = calloc(count + 1u, sizeof *si);
    VkTimelineSemaphoreSubmitInfo* ti = calloc(count + 1u, sizeof *ti);
    VkSemaphore* semaphores = calloc(waits + signals + 1, sizeof *semaphores);
    uint64_t* values = calloc(waits + signals + 1, sizeof *values);
    VkPipelineStageFlags* stages = calloc(waits + 1, sizeof *stages);
    VkCommandBuffer* cbs = calloc(buffers + 1, sizeof *cbs);
    if (!si || !ti || !semaphores || !values || !stages || !cbs) {
        free(si), free(ti), free(semaphores), free(values), free(stages), free(cbs);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    size_t at_sem = 0, at_stage = 0, at_cb = 0;
    for (uint32_t i = 0; i < count; i++) {
        const VkSubmitInfo2* s = &in[i];
        const size_t first_wait = at_sem;
        for (uint32_t k = 0; k < s->waitSemaphoreInfoCount; k++, at_sem++, at_stage++) {
            const VkSemaphoreSubmitInfo* w = &s->pWaitSemaphoreInfos[k];
            semaphores[at_sem] = w->semaphore;
            values[at_sem] = w->value;
            /* a wait that must stop no stage cannot be said in 1.0: it stops all of them */
            const VkPipelineStageFlags st = stage_v1(w->stageMask);
            stages[at_stage] = st ? st : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        const size_t first_signal = at_sem;
        for (uint32_t k = 0; k < s->signalSemaphoreInfoCount; k++, at_sem++) {
            semaphores[at_sem] = s->pSignalSemaphoreInfos[k].semaphore;
            values[at_sem] = s->pSignalSemaphoreInfos[k].value;
        }
        const size_t first_cb = at_cb;
        for (uint32_t k = 0; k < s->commandBufferInfoCount; k++) cbs[at_cb++] = s->pCommandBufferInfos[k].commandBuffer;
        ti[i] = (VkTimelineSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                                .pNext = s->pNext,
                                                .waitSemaphoreValueCount = s->waitSemaphoreInfoCount,
                                                .pWaitSemaphoreValues = values + first_wait,
                                                .signalSemaphoreValueCount = s->signalSemaphoreInfoCount,
                                                .pSignalSemaphoreValues = values + first_signal};
        si[i] = (VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                               .pNext = &ti[i],
                               .waitSemaphoreCount = s->waitSemaphoreInfoCount,
                               .pWaitSemaphores = semaphores + first_wait,
                               .pWaitDstStageMask = stages + (at_stage - s->waitSemaphoreInfoCount),
                               .commandBufferCount = s->commandBufferInfoCount,
                               .pCommandBuffers = cbs + first_cb,
                               .signalSemaphoreCount = s->signalSemaphoreInfoCount,
                               .pSignalSemaphores = semaphores + first_signal};
    }
    const VkResult r = vkQueueSubmit(queue, count, si, fence);
    free(si), free(ti), free(semaphores), free(values), free(stages), free(cbs);
    return r;
}

static void VKAPI_CALL compat_cmd_write_timestamp2(VkCommandBuffer cb, VkPipelineStageFlags2 stage, VkQueryPool pool,
                                                   uint32_t query) {
    VkPipelineStageFlags s = stage_v1(stage);
    /* one stage: none is the top of the pipe, and two or more are all of them */
    if (!s) s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    else if (s & (s - 1)) s = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    vkCmdWriteTimestamp(cb, (VkPipelineStageFlagBits)s, pool, query);
}

void vk_sync2_compat_install(void) {
    vkCmdPipelineBarrier2 = compat_cmd_pipeline_barrier2;
    vkQueueSubmit2 = compat_queue_submit2;
    vkCmdWriteTimestamp2 = compat_cmd_write_timestamp2;
}
