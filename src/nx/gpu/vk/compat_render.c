/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* dynamic rendering for a device without it (compat_render.h). The design follows Eden's render pass cache
   (video_core/renderer_vulkan/vk_render_pass_cache.cpp, GPL-2.0-or-later, Copyright yuzu Emulator Project and Eden
   Emulator Project): a render pass for each set of formats, sample count, load and store ops and layouts, and one
   subpass. What is different here is where the set comes from: a vkCmdBeginRendering call carries only image views,
   so the calls that make images and views are wrapped to keep their formats and sizes.

   - vkCmdBeginRendering finds or makes the render pass and the framebuffer of its views, and begins the pass.
     The layout of each attachment is the same at the start, in the pass and at the end: the barriers outside the pass
     do the transitions, as they do with dynamic rendering.
   - vkCreateGraphicsPipelines gives each pipeline the render pass of the formats in its
     VkPipelineRenderingCreateInfo. That pass is compatible with the one a pass of the same formats begins with, as
     the load and store ops and the layouts are not part of the compatibility.
   - A framebuffer is kept until a view of it is destroyed. The renderer destroys a view only after the commands that
     use it are done, so the framebuffers of that view are done too.
   What does not exist here: a pass that is suspended or resumed, secondary command buffers, a shading rate
   attachment, and local reads of an attachment. The renderer does not use them on a device without them. */
#include "nx/gpu/vk/compat_render.h"
#include "common/utilities/bits.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { COLORS = 8 };

#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES == 1
#define HANDLE_U64(h) ((uint64_t)(uintptr_t)(h))
#define U64_HANDLE(type, v) ((type)(uintptr_t)(v))
#else
#define HANDLE_U64(h) ((uint64_t)(h))
#define U64_HANDLE(type, v) ((type)(v))
#endif

/* ---- the driver's calls, and the lock of what is kept ---- */

static PFN_vkCreateImage real_create_image;
static PFN_vkDestroyImage real_destroy_image;
static PFN_vkCreateImageView real_create_view;
static PFN_vkDestroyImageView real_destroy_view;
static PFN_vkCreateGraphicsPipelines real_create_pipelines;
static VkDevice g_device;
static atomic_flag g_lock = ATOMIC_FLAG_INIT;

static void lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_lock, memory_order_acquire)) {
    }
}
static void unlock(void) { atomic_flag_clear_explicit(&g_lock, memory_order_release); }

/* ---- the images and views that are known: handle -> what a pass is made from ---- */

typedef struct {
    uint64_t key; /* the handle, 0 for a free place */
    uint32_t format, samples, width, height, layers; /* of an image: no format; of a view: the size of its level */
    uint8_t attached;                                /* a framebuffer holds this view */
} Info;
typedef struct {
    Info* e;
    size_t cap, n;
} Map;
static Map g_images, g_views;

static size_t slot(uint64_t key, size_t mask) { return (size_t)((key * 0x9E3779B97F4A7C15ull) >> 24) & mask; }

static Info* map_find(Map* m, uint64_t key) {
    if (!m->cap || !key) return NULL;
    const size_t mask = m->cap - 1;
    for (size_t i = slot(key, mask);; i = (i + 1) & mask) {
        if (!m->e[i].key) return NULL;
        if (m->e[i].key == key) return &m->e[i];
    }
}
static void map_insert(Map* m, const Info* v) {
    const size_t mask = m->cap - 1;
    size_t i = slot(v->key, mask);
    while (m->e[i].key && m->e[i].key != v->key) i = (i + 1) & mask;
    if (!m->e[i].key) m->n++;
    m->e[i] = *v;
}
static void map_put(Map* m, const Info* v) {
    if ((m->n + 1) * 2 > m->cap) {
        const size_t cap = m->cap ? m->cap * 2 : 256;
        Info* e = calloc(cap, sizeof *e);
        if (!e) return; /* not known: a pass of it takes the sizes of its render area */
        Map bigger = {e, cap, 0};
        for (size_t i = 0; i < m->cap; i++)
            if (m->e[i].key) map_insert(&bigger, &m->e[i]);
        free(m->e);
        *m = bigger;
    }
    map_insert(m, v);
}
static void map_del(Map* m, uint64_t key) {
    Info* found = map_find(m, key);
    if (!found) return;
    const size_t mask = m->cap - 1;
    size_t i = (size_t)(found - m->e), j = i;
    for (;;) { /* backward shift: the places after i that a probe from their own slot passes i for move into it */
        j = (j + 1) & mask;
        if (!m->e[j].key) break;
        const size_t home = slot(m->e[j].key, mask);
        if (((j - home) & mask) >= ((j - i) & mask)) {
            m->e[i] = m->e[j];
            i = j;
        }
    }
    m->e[i] = (Info){0};
    m->n--;
}

/* ---- caches of render passes and framebuffers: a key of fixed size -> a handle ---- */

typedef struct {
    uint64_t hash, value; /* value 0: dropped, its place stays for the probes that pass it */
    unsigned char key[];
} Entry;
typedef struct {
    size_t key_size, stride, n, cap, table_cap;
    unsigned char* entries;
    int32_t* table; /* the index of an entry + 1, 0 for none */
} Cache;
static Cache g_passes = {.key_size = 0}, g_buffers = {.key_size = 0};

static Entry* entry_at(const Cache* c, size_t i) { return (Entry*)(void*)(c->entries + i * c->stride); }

static int cache_rebuild(Cache* c) {
    const size_t table_cap = c->table_cap ? c->table_cap * 2 : 1024;
    int32_t* table = calloc(table_cap, sizeof *table);
    if (!table) return 0;
    size_t live = 0;
    for (size_t i = 0; i < c->n; i++) {
        Entry* e = entry_at(c, i);
        if (!e->value) continue;
        if (live != i) memcpy(entry_at(c, live), e, c->stride);
        size_t at = (size_t)entry_at(c, live)->hash & (table_cap - 1);
        while (table[at]) at = (at + 1) & (table_cap - 1);
        table[at] = (int32_t)(live + 1);
        live++;
    }
    free(c->table);
    c->table = table;
    c->table_cap = table_cap;
    c->n = live;
    return 1;
}
static Entry* cache_find(const Cache* c, uint64_t hash, const void* key) {
    if (!c->table_cap) return NULL;
    for (size_t at = (size_t)hash & (c->table_cap - 1); c->table[at]; at = (at + 1) & (c->table_cap - 1)) {
        Entry* e = entry_at(c, (size_t)c->table[at] - 1);
        if (e->value && e->hash == hash && !memcmp(e->key, key, c->key_size)) return e;
    }
    return NULL;
}
/* the entry for key, made with value, or NULL where there is no memory */
static Entry* cache_add(Cache* c, uint64_t hash, const void* key, uint64_t value) {
    if (!c->stride) c->stride = (sizeof(Entry) + c->key_size + 7u) & ~(size_t)7u;
    if ((c->n + 1) * 2 > c->table_cap) {
        if (c->n == c->cap) {
            const size_t cap = c->cap ? c->cap * 2 : 512;
            unsigned char* grown = realloc(c->entries, cap * c->stride);
            if (!grown) return NULL;
            c->entries = grown;
            c->cap = cap;
        }
        if (!cache_rebuild(c)) return NULL;
    }
    if (c->n == c->cap) {
        const size_t cap = c->cap ? c->cap * 2 : 512;
        unsigned char* grown = realloc(c->entries, cap * c->stride);
        if (!grown) return NULL;
        c->entries = grown;
        c->cap = cap;
    }
    Entry* e = entry_at(c, c->n);
    e->hash = hash;
    e->value = value;
    memcpy(e->key, key, c->key_size);
    size_t at = (size_t)hash & (c->table_cap - 1);
    while (c->table[at]) at = (at + 1) & (c->table_cap - 1);
    c->table[at] = (int32_t)(++c->n);
    return e;
}
static void cache_free(Cache* c) {
    free(c->entries);
    free(c->table);
    *c = (Cache){.key_size = c->key_size};
}

/* ---- render passes ---- */

/* what a render pass is made from. Every member is 32 bits, and the whole is zeroed before it is filled, so that two
   keys that mean the same hold the same bytes */
typedef struct {
    uint32_t view_mask, samples;
    uint32_t color_format[COLORS], color_load[COLORS], color_store[COLORS], color_layout[COLORS];
    uint32_t resolve_format[COLORS], resolve_layout[COLORS];
    uint32_t depth_format, depth_load, depth_store, stencil_load, stencil_store, depth_layout;
    uint32_t depth_resolve_format, depth_resolve_layout, depth_resolve_mode;
} PassKey;

static int has_stencil(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           f == VK_FORMAT_S8_UINT;
}

static VkAttachmentDescription2 describe(uint32_t format, uint32_t samples, uint32_t load, uint32_t store,
                                         uint32_t stencil_load, uint32_t stencil_store, uint32_t layout) {
    return (VkAttachmentDescription2){.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
                                      .format = (VkFormat)format,
                                      .samples = (VkSampleCountFlagBits)samples,
                                      .loadOp = (VkAttachmentLoadOp)load,
                                      .storeOp = (VkAttachmentStoreOp)store,
                                      .stencilLoadOp = (VkAttachmentLoadOp)stencil_load,
                                      .stencilStoreOp = (VkAttachmentStoreOp)stencil_store,
                                      .initialLayout = (VkImageLayout)layout,
                                      .finalLayout = (VkImageLayout)layout};
}
static VkAttachmentReference2 refer(uint32_t attachment, uint32_t layout) {
    return (VkAttachmentReference2){.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
                                    .attachment = attachment,
                                    .layout = (VkImageLayout)layout};
}

/* the attachments in this order: the colors, the depth and stencil, the color resolves, the depth and stencil resolve.
   plan_pass lists the views in the same order */
static VkRenderPass make_pass(const PassKey* k) {
    VkAttachmentDescription2 at[COLORS * 2 + 2];
    VkAttachmentReference2 colors[COLORS], resolves[COLORS], depth, depth_resolve;
    uint32_t n = 0, color_count = 0, resolve_any = 0;
    for (uint32_t i = 0; i < COLORS; i++) {
        if (!k->color_format[i]) {
            colors[i] = refer(VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED);
            continue;
        }
        at[n] = describe(k->color_format[i], k->samples, k->color_load[i], k->color_store[i],
                         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, k->color_layout[i]);
        colors[i] = refer(n++, k->color_layout[i]);
        color_count = i + 1;
    }
    const VkFormat df = (VkFormat)k->depth_format;
    if (df) {
        at[n] = describe(k->depth_format, k->samples, k->depth_load, k->depth_store, k->stencil_load, k->stencil_store,
                         k->depth_layout);
        depth = refer(n++, k->depth_layout);
    }
    for (uint32_t i = 0; i < COLORS; i++) {
        if (!k->resolve_format[i]) {
            resolves[i] = refer(VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED);
            continue;
        }
        at[n] = describe(k->resolve_format[i], VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                         VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE,
                         k->resolve_layout[i]);
        resolves[i] = refer(n++, k->resolve_layout[i]);
        resolve_any = 1;
    }
    if (k->depth_resolve_format) {
        at[n] = describe(k->depth_resolve_format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                         VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE,
                         k->depth_resolve_layout);
        depth_resolve = refer(n++, k->depth_resolve_layout);
    }
    const VkSubpassDescriptionDepthStencilResolve ds_resolve = {
        .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
        .depthResolveMode = (VkResolveModeFlagBits)k->depth_resolve_mode,
        .stencilResolveMode =
            has_stencil((VkFormat)k->depth_resolve_format) ? (VkResolveModeFlagBits)k->depth_resolve_mode
                                                           : VK_RESOLVE_MODE_NONE,
        .pDepthStencilResolveAttachment = &depth_resolve};
    const VkSubpassDescription2 subpass = {.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
                                           .pNext = k->depth_resolve_format ? &ds_resolve : NULL,
                                           .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                           .viewMask = k->view_mask,
                                           .colorAttachmentCount = color_count,
                                           .pColorAttachments = colors,
                                           .pResolveAttachments = resolve_any ? resolves : NULL,
                                           .pDepthStencilAttachment = df ? &depth : NULL};
    const VkRenderPassCreateInfo2 ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
                                        .attachmentCount = n,
                                        .pAttachments = n ? at : NULL,
                                        .subpassCount = 1,
                                        .pSubpasses = &subpass};
    VkRenderPass pass = VK_NULL_HANDLE;
    if (vkCreateRenderPass2(g_device, &ci, NULL, &pass) != VK_SUCCESS) return VK_NULL_HANDLE;
    return pass;
}

static uint64_t hash_bytes(const void* p, size_t n) { return snx_fnv1a(SNX_FNV_BASIS, p, n); }

/* the render pass of a key, made once. The caller holds the lock */
static VkRenderPass pass_for(const PassKey* k) {
    const uint64_t hash = hash_bytes(k, sizeof *k);
    Entry* e = cache_find(&g_passes, hash, k);
    if (e) return U64_HANDLE(VkRenderPass, e->value);
    VkRenderPass pass = make_pass(k);
    if (!pass) return VK_NULL_HANDLE;
    if (!cache_add(&g_passes, hash, k, HANDLE_U64(pass))) {
        vkDestroyRenderPass(g_device, pass, NULL);
        return VK_NULL_HANDLE;
    }
    return pass;
}

/* ---- vkCmdBeginRendering and vkCmdEndRendering ---- */

typedef struct {
    uint64_t render_pass; /* the handle */
    uint64_t view[COLORS * 2 + 2];
    uint32_t width, height, layers;
    uint32_t pad;
} BufferKey;

/* what a pass of ri is: its key, its views in the order of make_pass, and its clear values by attachment. The sizes of
   the views that are known: *width and *height the least, or 0 where one is not (a swapchain's image) */
typedef struct {
    PassKey key;
    VkImageView views[COLORS * 2 + 2];
    VkClearValue clears[COLORS * 2 + 2];
    uint32_t count, width, height, layers;
} Plan;

static void measure(Plan* p, VkImageView view, int* unknown) {
    const Info* v = map_find(&g_views, HANDLE_U64(view));
    if (!v || !v->width || !v->height) {
        *unknown = 1;
        return;
    }
    if (!p->width || v->width < p->width) p->width = v->width;
    if (!p->height || v->height < p->height) p->height = v->height;
    if (v->layers && (!p->layers || v->layers < p->layers)) p->layers = v->layers;
}

/* false where a color attachment has no slot here. The caller holds the lock */
static int plan_pass(const VkRenderingInfo* ri, Plan* p) {
    *p = (Plan){0};
    p->key.view_mask = ri->viewMask;
    p->key.samples = VK_SAMPLE_COUNT_1_BIT;
    if (ri->colorAttachmentCount > COLORS) return 0;
    int unknown = 0, sampled = 0;
    const VkRenderingAttachmentInfo* d = ri->pDepthAttachment;
    const VkRenderingAttachmentInfo* s = ri->pStencilAttachment;
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) {
        const VkRenderingAttachmentInfo* a = &ri->pColorAttachments[i];
        const Info* v = a->imageView ? map_find(&g_views, HANDLE_U64(a->imageView)) : NULL;
        if (!a->imageView) continue;
        if (!v) return 0;
        p->key.color_format[i] = v->format;
        p->key.color_load[i] = (uint32_t)a->loadOp;
        p->key.color_store[i] = (uint32_t)a->storeOp;
        p->key.color_layout[i] = (uint32_t)a->imageLayout;
        p->views[p->count] = a->imageView;
        p->clears[p->count++].color = a->clearValue.color;
        measure(p, a->imageView, &unknown);
        if (!sampled && v->samples) {
            p->key.samples = v->samples;
            sampled = 1;
        }
    }
    const VkRenderingAttachmentInfo* z = d ? d : s;
    if (z && z->imageView) {
        const Info* v = map_find(&g_views, HANDLE_U64(z->imageView));
        if (!v) return 0;
        const int stencil = has_stencil((VkFormat)v->format);
        p->key.depth_format = v->format;
        p->key.depth_layout = (uint32_t)z->imageLayout;
        /* the aspect that is not attached keeps its contents: it is loaded and stored */
        p->key.depth_load = d ? (uint32_t)d->loadOp : (uint32_t)VK_ATTACHMENT_LOAD_OP_LOAD;
        p->key.depth_store = d ? (uint32_t)d->storeOp : (uint32_t)VK_ATTACHMENT_STORE_OP_STORE;
        p->key.stencil_load = !stencil ? (uint32_t)VK_ATTACHMENT_LOAD_OP_DONT_CARE
                              : s      ? (uint32_t)s->loadOp
                                       : (uint32_t)VK_ATTACHMENT_LOAD_OP_LOAD;
        p->key.stencil_store = !stencil ? (uint32_t)VK_ATTACHMENT_STORE_OP_DONT_CARE
                               : s      ? (uint32_t)s->storeOp
                                        : (uint32_t)VK_ATTACHMENT_STORE_OP_STORE;
        p->views[p->count] = z->imageView;
        if (d) p->clears[p->count].depthStencil.depth = d->clearValue.depthStencil.depth;
        if (s) p->clears[p->count].depthStencil.stencil = s->clearValue.depthStencil.stencil;
        p->count++;
        measure(p, z->imageView, &unknown);
        if (!sampled && v->samples) {
            p->key.samples = v->samples;
            sampled = 1;
        }
    }
    for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) {
        const VkRenderingAttachmentInfo* a = &ri->pColorAttachments[i];
        if (!a->imageView || a->resolveMode == VK_RESOLVE_MODE_NONE || !a->resolveImageView) continue;
        const Info* v = map_find(&g_views, HANDLE_U64(a->resolveImageView));
        if (!v) return 0;
        p->key.resolve_format[i] = v->format;
        p->key.resolve_layout[i] = (uint32_t)a->resolveImageLayout;
        p->views[p->count++] = a->resolveImageView;
        measure(p, a->resolveImageView, &unknown);
    }
    const VkRenderingAttachmentInfo* r = d && d->resolveMode != VK_RESOLVE_MODE_NONE && d->resolveImageView ? d
                                         : s && s->resolveMode != VK_RESOLVE_MODE_NONE && s->resolveImageView ? s
                                                                                                              : NULL;
    if (r && z && z->imageView) {
        const Info* v = map_find(&g_views, HANDLE_U64(r->resolveImageView));
        if (!v) return 0;
        p->key.depth_resolve_format = v->format;
        p->key.depth_resolve_layout = (uint32_t)r->resolveImageLayout;
        p->key.depth_resolve_mode = (uint32_t)r->resolveMode;
        p->views[p->count++] = r->resolveImageView;
        measure(p, r->resolveImageView, &unknown);
    }
    /* a view of an image that is not known (a swapchain's): the render area's end is a size that every view has */
    if (unknown) p->width = p->height = 0;
    return 1;
}

static void VKAPI_CALL compat_cmd_begin_rendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
    Plan plan;
    VkRenderPass pass = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    lock();
    if (plan_pass(ri, &plan)) {
        pass = pass_for(&plan.key);
        if (pass) {
            BufferKey key;
            memset(&key, 0, sizeof key);
            key.render_pass = HANDLE_U64(pass);
            for (uint32_t i = 0; i < plan.count; i++) key.view[i] = HANDLE_U64(plan.views[i]);
            key.width = plan.width ? plan.width : (uint32_t)ri->renderArea.offset.x + ri->renderArea.extent.width;
            key.height = plan.height ? plan.height : (uint32_t)ri->renderArea.offset.y + ri->renderArea.extent.height;
            /* a pass with views begins on their layers: a multiview pass has one layer, as a framebuffer's */
            key.layers = ri->viewMask ? 1u : (ri->layerCount ? ri->layerCount : 1u);
            const uint64_t hash = hash_bytes(&key, sizeof key);
            const Entry* e = cache_find(&g_buffers, hash, &key);
            if (e) {
                fb = U64_HANDLE(VkFramebuffer, e->value);
            } else {
                const VkFramebufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                                    .renderPass = pass,
                                                    .attachmentCount = plan.count,
                                                    .pAttachments = plan.count ? plan.views : NULL,
                                                    .width = key.width,
                                                    .height = key.height,
                                                    .layers = key.layers};
                if (vkCreateFramebuffer(g_device, &ci, NULL, &fb) == VK_SUCCESS) {
                    if (cache_add(&g_buffers, hash, &key, HANDLE_U64(fb))) {
                        for (uint32_t i = 0; i < plan.count; i++) {
                            Info* v = map_find(&g_views, HANDLE_U64(plan.views[i]));
                            if (v) v->attached = 1;
                        }
                    } else {
                        vkDestroyFramebuffer(g_device, fb, NULL);
                        fb = VK_NULL_HANDLE;
                    }
                } else {
                    fb = VK_NULL_HANDLE;
                }
            }
        }
    }
    unlock();
    if (!fb) return; /* nothing to draw into: the draws after it are the renderer's own errors to report */
    const VkRenderPassBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      .renderPass = pass,
                                      .framebuffer = fb,
                                      .renderArea = ri->renderArea,
                                      .clearValueCount = plan.count,
                                      .pClearValues = plan.clears};
    vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
}

static void VKAPI_CALL compat_cmd_end_rendering(VkCommandBuffer cb) { vkCmdEndRenderPass(cb); }

/* ---- images and views ---- */

static VkResult VKAPI_CALL wrap_create_image(VkDevice device, const VkImageCreateInfo* ci,
                                             const VkAllocationCallbacks* alloc, VkImage* image) {
    const VkResult r = real_create_image(device, ci, alloc, image);
    if (r == VK_SUCCESS) {
        const Info v = {.key = HANDLE_U64(*image),
                        .samples = (uint32_t)ci->samples,
                        .width = ci->extent.width,
                        .height = ci->extent.height,
                        .layers = ci->arrayLayers};
        lock();
        map_put(&g_images, &v);
        unlock();
    }
    return r;
}
static void VKAPI_CALL wrap_destroy_image(VkDevice device, VkImage image, const VkAllocationCallbacks* alloc) {
    lock();
    map_del(&g_images, HANDLE_U64(image));
    unlock();
    real_destroy_image(device, image, alloc);
}

/* the framebuffers that hold a view are destroyed: it is going away. The caller holds the lock */
static void drop_buffers_of(VkImageView view) {
    const uint64_t handle = HANDLE_U64(view);
    for (size_t i = 0; i < g_buffers.n; i++) {
        Entry* e = entry_at(&g_buffers, i);
        if (!e->value) continue;
        const BufferKey* key = (const BufferKey*)(const void*)e->key;
        for (size_t k = 0; k < sizeof key->view / sizeof *key->view; k++) {
            if (key->view[k] != handle) continue;
            vkDestroyFramebuffer(g_device, U64_HANDLE(VkFramebuffer, e->value), NULL);
            e->value = 0;
            break;
        }
    }
}

static VkResult VKAPI_CALL wrap_create_view(VkDevice device, const VkImageViewCreateInfo* ci,
                                            const VkAllocationCallbacks* alloc, VkImageView* view) {
    const VkResult r = real_create_view(device, ci, alloc, view);
    if (r == VK_SUCCESS) {
        Info v = {.key = HANDLE_U64(*view), .format = (uint32_t)ci->format};
        lock();
        const Info* img = map_find(&g_images, HANDLE_U64(ci->image));
        if (img) {
            const uint32_t level = ci->subresourceRange.baseMipLevel;
            v.samples = img->samples;
            v.width = img->width >> level ? img->width >> level : 1u;
            v.height = img->height >> level ? img->height >> level : 1u;
            const uint32_t left = img->layers > ci->subresourceRange.baseArrayLayer
                                      ? img->layers - ci->subresourceRange.baseArrayLayer
                                      : 1u;
            v.layers = ci->subresourceRange.layerCount == VK_REMAINING_ARRAY_LAYERS
                           ? left
                           : snx_min_u32(ci->subresourceRange.layerCount, left);
        }
        map_put(&g_views, &v);
        unlock();
    }
    return r;
}
static void VKAPI_CALL wrap_destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks* alloc) {
    lock();
    const Info* v = map_find(&g_views, HANDLE_U64(view));
    if (v && v->attached) drop_buffers_of(view);
    map_del(&g_views, HANDLE_U64(view));
    unlock();
    real_destroy_view(device, view, alloc);
}

/* ---- pipelines ---- */

static VkResult VKAPI_CALL wrap_create_pipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                 const VkGraphicsPipelineCreateInfo* in,
                                                 const VkAllocationCallbacks* alloc, VkPipeline* out) {
    VkGraphicsPipelineCreateInfo small[8];
    VkGraphicsPipelineCreateInfo* ci = count <= 8 ? small : malloc(count * sizeof *ci);
    if (!ci) return VK_ERROR_OUT_OF_HOST_MEMORY;
    memcpy(ci, in, count * sizeof *ci);
    for (uint32_t i = 0; i < count; i++) {
        if (ci[i].renderPass) continue;
        const VkPipelineRenderingCreateInfo* r = NULL;
        for (const VkBaseInStructure* b = ci[i].pNext; b; b = b->pNext)
            if (b->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) r = (const void*)b;
        if (!r) continue;
        if (r->colorAttachmentCount > COLORS) {
            if (ci != small) free(ci);
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        PassKey key;
        memset(&key, 0, sizeof key);
        key.view_mask = r->viewMask;
        key.samples = ci[i].pMultisampleState ? (uint32_t)ci[i].pMultisampleState->rasterizationSamples
                                              : (uint32_t)VK_SAMPLE_COUNT_1_BIT;
        for (uint32_t k = 0; k < r->colorAttachmentCount; k++) {
            key.color_format[k] = (uint32_t)r->pColorAttachmentFormats[k];
            key.color_load[k] = VK_ATTACHMENT_LOAD_OP_LOAD;
            key.color_store[k] = VK_ATTACHMENT_STORE_OP_STORE;
            key.color_layout[k] = VK_IMAGE_LAYOUT_GENERAL;
        }
        const VkFormat depth = r->depthAttachmentFormat != VK_FORMAT_UNDEFINED ? r->depthAttachmentFormat
                                                                               : r->stencilAttachmentFormat;
        if (depth != VK_FORMAT_UNDEFINED) {
            key.depth_format = (uint32_t)depth;
            key.depth_load = VK_ATTACHMENT_LOAD_OP_LOAD;
            key.depth_store = VK_ATTACHMENT_STORE_OP_STORE;
            key.stencil_load = has_stencil(depth) ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            key.stencil_store = has_stencil(depth) ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            key.depth_layout = VK_IMAGE_LAYOUT_GENERAL;
        }
        lock();
        ci[i].renderPass = pass_for(&key);
        unlock();
        ci[i].subpass = 0;
        if (!ci[i].renderPass) {
            if (ci != small) free(ci);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    const VkResult r = real_create_pipelines(device, cache, count, ci, alloc, out);
    if (ci != small) free(ci);
    return r;
}

/* ---- the install ---- */

void vk_render_compat_install(VkDevice dev) {
    /* what a device before this one kept is of that device: the objects went with it */
    free(g_images.e);
    free(g_views.e);
    g_images = (Map){0};
    g_views = (Map){0};
    cache_free(&g_passes);
    cache_free(&g_buffers);
    g_passes.key_size = sizeof(PassKey);
    g_buffers.key_size = sizeof(BufferKey);
    g_device = dev;
    /* the driver's calls, where volk has just loaded them: not our own, where this runs for a device again */
    if (vkCreateImage != wrap_create_image) real_create_image = vkCreateImage;
    if (vkDestroyImage != wrap_destroy_image) real_destroy_image = vkDestroyImage;
    if (vkCreateImageView != wrap_create_view) real_create_view = vkCreateImageView;
    if (vkDestroyImageView != wrap_destroy_view) real_destroy_view = vkDestroyImageView;
    if (vkCreateGraphicsPipelines != wrap_create_pipelines) real_create_pipelines = vkCreateGraphicsPipelines;
    vkCreateImage = wrap_create_image;
    vkDestroyImage = wrap_destroy_image;
    vkCreateImageView = wrap_create_view;
    vkDestroyImageView = wrap_destroy_view;
    vkCreateGraphicsPipelines = wrap_create_pipelines;
    vkCmdBeginRendering = compat_cmd_begin_rendering;
    vkCmdEndRendering = compat_cmd_end_rendering;
}
