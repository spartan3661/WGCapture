#define NOMINMAX
#include <Windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <thread>
#include <winrt/base.h>
#include <winrt/windows.graphics.capture.h>
#include <winrt/windows.graphics.directx.direct3d11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include "wgc_interop.h"
#include "wgc_capture.h"
#include "wgc_api.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

using Microsoft::WRL::ComPtr;

static void* heap_alloc(size_t n) { return ::CoTaskMemAlloc(n); }
static void  heap_free(void* p) { if (p) ::CoTaskMemFree(p); }

extern "C" __declspec(dllexport) void __cdecl wgc_free(void* p) { heap_free(p); }

static bool copy_to_host(ID3D11Device* dev, ID3D11DeviceContext* ctx,
    ID3D11Texture2D* src,
    uint8_t** out_data, int* out_size, int* out_w, int* out_h, int* out_stride)
{
    D3D11_TEXTURE2D_DESC d{};
    src->GetDesc(&d);
    d.BindFlags = 0;
    d.MiscFlags = 0;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(dev->CreateTexture2D(&d, nullptr, &staging))) return false;

    ctx->CopyResource(staging.Get(), src);

    D3D11_MAPPED_SUBRESOURCE m{};
    if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;

    const int w = (int)d.Width;
    const int h = (int)d.Height;
    const int stride = (int)m.RowPitch;
    const int size = stride * h;

    uint8_t* buf = (uint8_t*)heap_alloc(size);
    if (!buf) { ctx->Unmap(staging.Get(), 0); return false; }

    uint8_t* dst = buf;
    uint8_t* srcp = (uint8_t*)m.pData;
    for (int y = 0; y < h; ++y) {
        memcpy(dst, srcp, stride);
        dst += stride;
        srcp += m.RowPitch;
    }

    ctx->Unmap(staging.Get(), 0);

    *out_data = buf;
    *out_size = size;
    *out_w = w;
    *out_h = h;
    *out_stride = stride;
    return true;
}

static int capture_one_frame_on_this_thread(HWND hwnd, wgc_bgra_frame* out, int timeout_ms)
{
    using namespace winrt;

    // worker lets us safely call .get() on WinRT async ops
    init_apartment(apartment_type::multi_threaded);

    ComPtr<ID3D11Device> d3d;
    ComPtr<ID3D11DeviceContext> ctx;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &d3d, nullptr, &ctx))) {
        return WGC_ERR_START;
    }

    WgcCapture cap;
    if (!cap.StartWinCap(d3d.Get(), hwnd, /*cursor*/false, /*borderless*/true))
        return WGC_ERR_START;

    const ULONGLONG deadline = GetTickCount64() + (ULONGLONG)(timeout_ms > 0 ? timeout_ms : 2000);
    ID3D11Texture2D* tex = nullptr;

    while (GetTickCount64() < deadline) {
        if (cap.GrabNextFrame()) {
            tex = cap.LatestTexture();
            if (tex) break;
        }
        ::Sleep(5);
    }

    if (!tex) { cap.Stop(); return WGC_ERR_TIMEOUT; }

    uint8_t* data = nullptr; int size = 0, w = 0, h = 0, stride = 0;
    if (!copy_to_host(d3d.Get(), ctx.Get(), tex, &data, &size, &w, &h, &stride)) {
        cap.Stop(); return WGC_ERR_COPY;
    }

    cap.Stop();

    out->data = data;
    out->size = size;
    out->width = w;
    out->height = h;
    out->stride = stride;
    return WGC_OK;
}

extern "C" __declspec(dllexport)
int __cdecl wgc_capture_bgra(void* hwnd_void, struct wgc_bgra_frame* out, int timeout_ms)
{
    try {
        if (!hwnd_void || !out) return WGC_ERR_INVALID_ARG;
        HWND hwnd = (HWND)hwnd_void;

        int rc = WGC_ERR_START;
        // Run synchronously on separate worker
        std::thread worker([&]() {
            try {
                rc = capture_one_frame_on_this_thread(hwnd, out, timeout_ms);
            }
            catch (...) {
                rc = WGC_ERR_START;
            }
            });
        worker.join();
        return rc;
    }
    catch (...) {
        return WGC_ERR_START;
    }
}
