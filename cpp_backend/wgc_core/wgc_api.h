#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    enum wgc_status {
        WGC_OK = 0,
        WGC_ERR_INVALID_ARG = -1,
        WGC_ERR_START = -2,
        WGC_ERR_TIMEOUT = -3,
        WGC_ERR_COPY = -4,
        WGC_ERR_ENCODE = -5
    };

    struct wgc_bgra_frame {
        uint8_t* data;
        int32_t  width;
        int32_t  height;
        int32_t  stride;
        int32_t  size;
    };

    __declspec(dllexport) int  __cdecl wgc_capture_bgra(void* hwnd, struct wgc_bgra_frame* out, int timeout_ms);
    __declspec(dllexport) void __cdecl wgc_free(void* p);

#ifdef __cplusplus
}
#endif
