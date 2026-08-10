from windows.com import IUnknown, HRESULT
import guid

# IPersist - objidl.h. About as minimal as a real, standard COM
# interface gets: IUnknown's 3 slots plus exactly one method,
# GetClassID. Signature verified against Microsoft Learn's own
# IPersist::GetClassID docs, not just memory (see
# PLAN_SUBCLASSING_VTABLES_COM.md's own note on why real-COM-interop
# GUIDs/signatures got this scrutiny - a wrong vtable shape here calls
# into whatever real function actually sits at that offset, not a
# graceful failure).
@interface
class IPersist( IUnknown ):
	@virtual
	def GetClassID( self, pClassID: Ptr[guid.GUID] ) -> HRESULT: ...
