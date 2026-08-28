'''
Minimal D3D11 (d3d11.dll) surface - just enough to create a device, make a
CPU-readable staging texture, and copy/map a captured desktop frame into
it. Vtable order and every struct/GUID here verified directly against
this machine's own Windows SDK header (d3d11.h) - see windows/com/
idxgi.py's own header comment for the same "real SDK header, not memory"
scrutiny and the "stub slots hold real offsets" convention this follows.
'''

from windows.com import IUnknown, HRESULT
import guid

def _guid( data1: u32, data2: u16, data3: u16, d0: u8, d1: u8, d2: u8, d3: u8, d4: u8, d5: u8, d6: u8, d7: u8 ) -> guid.GUID:
	g: guid.GUID = guid.GUID( data1 = data1, data2 = data2, data3 = data3, data4 = 0 )
	g.data4[0] = d0
	g.data4[1] = d1
	g.data4[2] = d2
	g.data4[3] = d3
	g.data4[4] = d4
	g.data4[5] = d5
	g.data4[6] = d6
	g.data4[7] = d7
	return g

# verified against this machine's d3d11.h DEFINE_GUID line directly
IID_ID3D11TEXTURE2D: guid.GUID = _guid( 0x6f15aaf2, 0xd208, 0x4e89, 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c )

# ---------------------------------------------------------------------------
# constants actually used by this project's capture path (d3d11.h/
# d3dcommon.h/dxgiformat.h) - not attempting a full transcription
# ---------------------------------------------------------------------------

D3D_DRIVER_TYPE_HARDWARE: u32 = 1     # D3D_DRIVER_TYPE_UNKNOWN(0) + 1
D3D11_SDK_VERSION: u32 = 7
D3D11_USAGE_STAGING: u32 = 3
D3D11_CPU_ACCESS_READ: u32 = 0x20000
D3D11_MAP_READ: u32 = 1
DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87

# ---------------------------------------------------------------------------
# plain value structs (d3d11.h)
# ---------------------------------------------------------------------------

@cstruct
class DXGI_SAMPLE_DESC:
	count: u32 = 1
	quality: u32 = 0

@cstruct
class D3D11_TEXTURE2D_DESC:
	width: u32 = 0
	height: u32 = 0
	mip_levels: u32 = 0
	array_size: u32 = 0
	format: u32 = 0
	sample_desc: DXGI_SAMPLE_DESC = DXGI_SAMPLE_DESC()
	usage: u32 = 0
	bind_flags: u32 = 0
	cpu_access_flags: u32 = 0
	misc_flags: u32 = 0

@cstruct
class D3D11_MAPPED_SUBRESOURCE:
	pData: Ptr[None] = None
	RowPitch: u32 = 0
	DepthPitch: u32 = 0

# ---------------------------------------------------------------------------
# D3D11CreateDevice - d3d11.dll's own entry point
# ---------------------------------------------------------------------------

@extern( 'd3d11', 'D3D11CreateDevice' )
def D3D11CreateDevice(
	pAdapter: Ptr[None],
	DriverType: u32,
	Software: Ptr[None],
	Flags: u32,
	pFeatureLevels: Ptr[None],
	FeatureLevels: u32,
	SDKVersion: u32,
	ppDevice: Ptr[Ptr[None]],
	pFeatureLevel: Ptr[u32],
	ppImmediateContext: Ptr[Ptr[None]],
) -> HRESULT:
	...

# ---------------------------------------------------------------------------
# ID3D11Device : IUnknown - only CreateTexture2D (3rd of its own methods)
# is called; CreateBuffer/CreateTexture1D are unused stubs kept only to
# hold CreateTexture2D at its real offset.
# ---------------------------------------------------------------------------

@interface
class ID3D11Device( IUnknown ):
	@virtual
	def CreateBuffer( self ) -> None: ... # unused stub
	@virtual
	def CreateTexture1D( self ) -> None: ... # unused stub

	@virtual
	def CreateTexture2D( self, pDesc: ConstPtr[D3D11_TEXTURE2D_DESC], pInitialData: Ptr[None], ppTexture2D: Ptr[Ptr[None]] ) -> HRESULT: ...

# ---------------------------------------------------------------------------
# ID3D11DeviceContext : ID3D11DeviceChild : IUnknown - Map/Unmap/
# CopyResource are called; everything else up through CopyResource
# (ID3D11DeviceChild's own 4, then most of ID3D11DeviceContext's own
# list) is unused stubs kept only to hold these three at their real
# offsets. Nothing declared past CopyResource - never called.
# ---------------------------------------------------------------------------

@interface
class ID3D11DeviceContext( IUnknown ):
	# ID3D11DeviceChild's own 4 methods
	@virtual
	def GetDevice( self ) -> None: ... # unused stub
	@virtual
	def GetPrivateData( self ) -> None: ... # unused stub
	@virtual
	def SetPrivateData( self ) -> None: ... # unused stub
	@virtual
	def SetPrivateDataInterface( self ) -> None: ... # unused stub

	# ID3D11DeviceContext's own methods, in real order, up through CopyResource
	@virtual
	def VSSetConstantBuffers( self ) -> None: ... # unused stub
	@virtual
	def PSSetShaderResources( self ) -> None: ... # unused stub
	@virtual
	def PSSetShader( self ) -> None: ... # unused stub
	@virtual
	def PSSetSamplers( self ) -> None: ... # unused stub
	@virtual
	def VSSetShader( self ) -> None: ... # unused stub
	@virtual
	def DrawIndexed( self ) -> None: ... # unused stub
	@virtual
	def Draw( self ) -> None: ... # unused stub

	@virtual
	def Map( self, pResource: Ptr[None], Subresource: u32, MapType: u32, MapFlags: u32, pMappedResource: Ptr[D3D11_MAPPED_SUBRESOURCE] ) -> HRESULT: ...
	@virtual
	def Unmap( self, pResource: Ptr[None], Subresource: u32 ) -> None: ...

	@virtual
	def PSSetConstantBuffers( self ) -> None: ... # unused stub
	@virtual
	def IASetInputLayout( self ) -> None: ... # unused stub
	@virtual
	def IASetVertexBuffers( self ) -> None: ... # unused stub
	@virtual
	def IASetIndexBuffer( self ) -> None: ... # unused stub
	@virtual
	def DrawIndexedInstanced( self ) -> None: ... # unused stub
	@virtual
	def DrawInstanced( self ) -> None: ... # unused stub
	@virtual
	def GSSetConstantBuffers( self ) -> None: ... # unused stub
	@virtual
	def GSSetShader( self ) -> None: ... # unused stub
	@virtual
	def IASetPrimitiveTopology( self ) -> None: ... # unused stub
	@virtual
	def VSSetShaderResources( self ) -> None: ... # unused stub
	@virtual
	def VSSetSamplers( self ) -> None: ... # unused stub
	@virtual
	def Begin( self ) -> None: ... # unused stub
	@virtual
	def End( self ) -> None: ... # unused stub
	@virtual
	def GetData( self ) -> None: ... # unused stub
	@virtual
	def SetPredication( self ) -> None: ... # unused stub
	@virtual
	def GSSetShaderResources( self ) -> None: ... # unused stub
	@virtual
	def GSSetSamplers( self ) -> None: ... # unused stub
	@virtual
	def OMSetRenderTargets( self ) -> None: ... # unused stub
	@virtual
	def OMSetRenderTargetsAndUnorderedAccessViews( self ) -> None: ... # unused stub
	@virtual
	def OMSetBlendState( self ) -> None: ... # unused stub
	@virtual
	def OMSetDepthStencilState( self ) -> None: ... # unused stub
	@virtual
	def SOSetTargets( self ) -> None: ... # unused stub
	@virtual
	def DrawAuto( self ) -> None: ... # unused stub
	@virtual
	def DrawIndexedInstancedIndirect( self ) -> None: ... # unused stub
	@virtual
	def DrawInstancedIndirect( self ) -> None: ... # unused stub
	@virtual
	def Dispatch( self ) -> None: ... # unused stub
	@virtual
	def DispatchIndirect( self ) -> None: ... # unused stub
	@virtual
	def RSSetState( self ) -> None: ... # unused stub
	@virtual
	def RSSetViewports( self ) -> None: ... # unused stub
	@virtual
	def RSSetScissorRects( self ) -> None: ... # unused stub
	@virtual
	def CopySubresourceRegion( self ) -> None: ... # unused stub

	@virtual
	def CopyResource( self, pDstResource: Ptr[None], pSrcResource: Ptr[None] ) -> None: ...
