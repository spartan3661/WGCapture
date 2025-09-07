#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <winrt/base.h>
#include <winrt/windows.foundation.h>
#include <winrt/windows.graphics.capture.h>
#include <winrt/windows.graphics.directx.direct3d11.h>

#include "wgc_interop.h"

class WgcCapture {
public:
    WgcCapture() = default;
    ~WgcCapture();

    bool StartWinCap(ID3D11Device* d3dDevice, HWND hwnd,
        bool captureCursor = false, bool borderless = true);
    void Stop();

    bool Active() const { return m_active; }

    // Pull the next frame on same thread that called StartWinCap.
    bool GrabNextFrame();

    ID3D11Texture2D* LatestTexture() const { return m_latest.Get(); }
    winrt::Windows::Graphics::SizeInt32 ContentSize() const { return m_lastSize; }

private:
    // D3D11
    Microsoft::WRL::ComPtr<ID3D11Device>        m_d3d;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_ctx;
    Microsoft::WRL::ComPtr<ID3D11Texture2D>     m_latest;

    // WinRT capture
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem          m_item{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool   m_framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession       m_session{ nullptr };
    winrt::Windows::Graphics::SizeInt32                             m_lastSize{ 0, 0 };

    bool m_active{ false };
    bool m_borderless{ true };
    bool m_captureCursor{ false };
};
