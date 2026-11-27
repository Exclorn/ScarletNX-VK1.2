/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The GL layer's sync calls on the console (gl_layer.h, gpu/gl/gl.hook's sync family): each the shared layer's
   (common/gpu/gl/gl_sync.h) on the calling thread's current context, under the object lock. A fence waits for the
   draws and clears before it, which the GPU thread finishes in order (gl_draw.c). A sync's handle is the layer's name, which a game only passes back.
   Without a current context a call does nothing and returns 0 */
#include "common/gpu/gl/gl_sync.h"
#include "gpu/gl/gl.hook.h"
#include "nx/gpu/gl/gl_layer.h"
#include "nx/gpu/nvn/nvn_object.h"
#include "nx/runtime/platform/log.h"
#include "nx/memory/mem.h"
#include <string.h>

static void say(const char* line) { WARN("gl layer: %s\n", line); }

/* NVN's calls that name GL objects (nvn_object.h's NvnGlInterop): a game that draws its interface through GL and the
   rest through NVN shares the interface's texture and its fences this way */
static int interop_texture(u32 name, NvnTexture* out) {
    gl_object_calls_lock();
    int ok = gl_nvn_texture(name, out);
    gl_object_calls_unlock();
    return ok;
}
static int interop_fenced(u64 sync) {
    gl_object_calls_lock();
    int ok = snx_gl_is_sync(sync);
    gl_object_calls_unlock();
    return ok;
}
static int interop_create(u64* sync) {
    u32 ctx = gl_egl_context(); /* the fence's errors go to a context: the calling thread's, else the first */
    gl_object_calls_lock();
    *sync = snx_gl_fence_sync(ctx ? ctx : 1, 0x9117 /* GL_SYNC_GPU_COMMANDS_COMPLETE */, 0);
    gl_object_calls_unlock();
    return *sync != 0;
}
void gl_sync_calls_init(void) {
    snx_gl_sync_host(&(SnxGlSyncHost){
        .recorded = gl_draw_recorded, .completed = gl_draw_completed, .wait = gl_draw_wait, .say = say});
    static const NvnGlInterop interop = {.texture = interop_texture, .fenced = interop_fenced, .create = interop_create};
    nvn_gl_interop(&interop);
}

#define LOCKED(fallback, ...)                                                                                          \
    do {                                                                                                               \
        u32 ctx = gl_egl_context();                                                                                    \
        (void)c;                                                                                                       \
        if (!ctx) return fallback;                                                                                     \
        gl_object_calls_lock();                                                                                        \
        __VA_ARGS__;                                                                                                   \
        gl_object_calls_unlock();                                                                                      \
    } while (0)
#define NONE

u64 gl_fence_sync(CPU* c, u32 condition, u32 flags) {
    u64 sync = 0;
    LOCKED(0, sync = snx_gl_fence_sync(ctx, condition, flags));
    return sync;
}
void gl_delete_sync(CPU* c, u64 sync) { LOCKED(NONE, snx_gl_delete_sync(ctx, sync)); }
u8 gl_is_sync(CPU* c, u64 sync) {
    u8 is = 0;
    LOCKED(0, is = (u8)snx_gl_is_sync(sync));
    return is;
}
u32 gl_client_wait_sync(CPU* c, u64 sync, u32 flags, u64 timeout) {
    u32 status = 0;
    LOCKED(0, status = snx_gl_client_wait_sync(ctx, sync, flags, timeout));
    return status;
}
void gl_wait_sync(CPU* c, u64 sync, u32 flags, u64 timeout) {
    LOCKED(NONE, snx_gl_wait_sync(ctx, sync, flags, timeout));
}
/* glGetSynciv(sync, pname, count, length, values) */
void gl_get_synciv(CPU* c, u64 sync, u32 pname, i32 count, u64 length, u64 values) {
    i32 v = 0, one = 1;
    LOCKED(
        NONE, if (snx_gl_get_sync(ctx, sync, pname, &v)) {
            if (count >= 1 && values) memcpy(GP(values), &v, 4);
            if (length) memcpy(GP(length), &one, 4);
        });
}
