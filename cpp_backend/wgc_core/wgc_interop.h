// wgc_interop.h
#pragma once
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/base.h>
#include <winrt/windows.foundation.h>
#include <winrt/windows.graphics.directx.direct3d11.h>

template<typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(
	winrt::Windows::Foundation::IInspectable const& obj)
{
	auto access = obj.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
	winrt::com_ptr<T> result;
	winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
	return result;
}