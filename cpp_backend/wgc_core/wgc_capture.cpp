#include "wgc_capture.h"
#include <windows.graphics.directx.direct3d11.interop.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.capture.h>
#include <windows.graphics.directx.h>

using namespace winrt;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;
using Microsoft::WRL::ComPtr;

WgcCapture::~WgcCapture() { Stop(); }

bool WgcCapture::StartWinCap(ID3D11Device* d3dDevice, HWND hwnd,
    bool captureCursor, bool borderless) {
    Stop();

    if (!d3dDevice || !hwnd) { return false; }
    m_captureCursor = captureCursor;
    m_borderless = borderless;

    // Hold D3D11 device/context
    m_d3d = d3dDevice;
    m_d3d->GetImmediateContext(&m_ctx);

    // Make WinRT IDirect3DDevice from DXGI
    ComPtr<IDXGIDevice> dxgi;
    if (FAILED(m_d3d.As(&dxgi))) return false;

    com_ptr<IInspectable> insp;
    if (FAILED(CreateDirect3D11DeviceFromDXGIDevice(dxgi.Get(), insp.put()))) return false;
    m_winrtDevice = insp.as<IDirect3DDevice>();

    // Create capture item using HWND
    auto factory = get_activation_factory<GraphicsCaptureItem>();
    auto interop = factory.as<IGraphicsCaptureItemInterop>();
    GraphicsCaptureItem item{ nullptr };
    const auto iid = winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>();
    if (FAILED(interop->CreateForWindow(hwnd, iid, reinterpret_cast<void**>(put_abi(item))))) return false;  // :contentReference[oaicite:1]{index=1}

    m_item = item;
    m_lastSize = m_item.Size();

    // .get() is OK only on free thread
    try {
        if (m_borderless) {
            GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless).get();  // :contentReference[oaicite:2]{index=2}
        }
    }
    catch (...) {}

    //prevent blocking single thread
    m_framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        m_winrtDevice,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        m_lastSize);  // :contentReference[oaicite:3]{index=3}

    m_session = m_framePool.CreateCaptureSession(m_item);

    try {
        if (m_borderless) {
            m_session.IsBorderRequired(false);
        }
    }
    catch (...) {}

    try {
        m_session.IsCursorCaptureEnabled(m_captureCursor);
    }
    catch (...) {}

    try {
        m_session.StartCapture();
        m_active = true;
        return true;
    }
    catch (...) {
        Stop();
        return false;
    }
}

void WgcCapture::Stop() {
    m_active = false;

    try { if (m_session)   m_session.Close(); }
    catch (...) {}
    try { if (m_framePool) m_framePool.Close(); }
    catch (...) {}

    m_session = nullptr;
    m_framePool = nullptr;
    m_item = nullptr;
    m_winrtDevice = nullptr;
    m_latest.Reset();
    m_ctx.Reset();
    m_d3d.Reset();
    m_lastSize = { 0,0 };
}

bool WgcCapture::GrabNextFrame() {
    if (!m_active || !m_framePool) return false;

    auto frame = m_framePool.TryGetNextFrame();
    if (!frame) return false;

    auto sz = frame.ContentSize();
    if (sz.Width != m_lastSize.Width || sz.Height != m_lastSize.Height) {
        m_lastSize = sz;
        try {
            m_framePool.Recreate(
                m_winrtDevice,
                winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                2, m_lastSize);
        }
        catch (...) { /* ignore */ }
    }

    // Unwrap surface -> ID3D11Texture2D
    winrt::com_ptr<ID3D11Texture2D> frameTex =
        GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());

    if (!frameTex) return false;

    // Ensure target texture of right size
    D3D11_TEXTURE2D_DESC desc{};
    frameTex->GetDesc(&desc);
    if (!m_latest ||
        desc.Width != static_cast<UINT>(m_lastSize.Width) ||
        desc.Height != static_cast<UINT>(m_lastSize.Height))
    {
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
        m_latest.Reset();
        if (FAILED(m_d3d->CreateTexture2D(&desc, nullptr, &m_latest))) return false;
    }

    // Copy frame
    m_ctx->CopyResource(m_latest.Get(), frameTex.get());
    return true;
}
