# lib/ssl.py — TLS/SSL client sockets. See PLAN_SSL.md for the backend-
# strategy decision (native OS TLS, no bundled OpenSSL) and phased roadmap.
#
# Client-side only (wrap an already-connected socket.Socket in a TLS
# session; no server side). Two backends so far: Windows (Schannel via
# SSPI, secur32.dll) and Linux (system OpenSSL via libssl, has_library-
# gated - requires libssl-dev on the build machine). macOS (Secure
# Transport / Network.framework) is still a later phase - see PLAN_SSL.md.

import compiler
import sys
import socket

@enum( i32 )
class SSLError:
	HandshakeFailed         = 1
	CertificateVerifyFailed = 2
	CertificateExpired      = 3
	HostnameMismatch        = 4
	ProtocolError            = 5
	Closed                   = 6
	Other = _


# ---------------------------------------------------------------------------
# Windows backend — SSPI/Schannel (secur32.dll). Struct layouts below were
# NOT hand-derived from memory: every sizeof()/offsetof() was cross-checked
# with a throwaway C probe compiled against the real Windows SDK headers
# (schannel.h/sspi.h via cl.exe), and the full handshake/encrypt/decrypt
# sequence was validated end-to-end with a standalone C client (real TCP +
# real Schannel handshake against example.com, and real cert-failure paths
# against badssl.com's expired/wrong-host/self-signed endpoints) before any
# of this was transcribed into MetalPy. See PLAN_SSL.md for the error-code
# mapping this uncovered (SEC_E_CERT_EXPIRED/SEC_E_WRONG_PRINCIPAL/
# SEC_E_UNTRUSTED_ROOT are real, distinct, confirmed-reachable codes).
#
# TypeAlias/const bindings live inside a plain module-level `if` (no `else`
# yet - Linux/macOS aren't implemented), matching lib/socket.py's own proven
# pattern for this (its own comment: "confirmed workable (a real compile) to
# declare @extern functions and TypeAlias/const bindings directly inside a
# module-level if/else" - @compiler.target(...) as a decorator is only
# established for def/class, never for a bare assignment).
# ---------------------------------------------------------------------------

if compiler.target.os == 'windows':
	SECURITY_STATUS: TypeAlias = i32
	TimeStamp: TypeAlias = i64  # LARGE_INTEGER-shaped; never read, only supplied

	# --- constants (schannel.h/sspi.h - well-known, ABI-stable values,
	# hardcoded rather than compiler.cexpr'd, matching
	# lib/windows/com/__init__.py's own precedent for winerror.h HRESULTs) ---

	SCHANNEL_CRED_VERSION:           u32 = 4
	SCH_CRED_NO_DEFAULT_CREDS:       u32 = 0x00000010
	SCH_CRED_AUTO_CRED_VALIDATION:   u32 = 0x00000020
	SCH_CRED_MANUAL_CRED_VALIDATION: u32 = 0x00000008

	SECPKG_CRED_OUTBOUND: u32 = 0x00000002

	ISC_REQ_SEQUENCE_DETECT:   u32 = 0x00000008
	ISC_REQ_REPLAY_DETECT:     u32 = 0x00000004
	ISC_REQ_CONFIDENTIALITY:   u32 = 0x00000010
	ISC_REQ_ALLOCATE_MEMORY:   u32 = 0x00000100
	ISC_REQ_STREAM:            u32 = 0x00008000
	ISC_RET_EXTRA_DATA:        u32 = 0x00000040

	SECBUFFER_VERSION:        u32 = 0
	SECBUFFER_EMPTY:          u32 = 0
	SECBUFFER_DATA:           u32 = 1
	SECBUFFER_TOKEN:          u32 = 2
	SECBUFFER_EXTRA:          u32 = 5
	SECBUFFER_STREAM_TRAILER: u32 = 6
	SECBUFFER_STREAM_HEADER:  u32 = 7

	SECURITY_NATIVE_DREP: u32 = 0x00000010

	SECPKG_ATTR_STREAM_SIZES: u32 = 4

	# high-bit-set SECURITY_STATUS values need an explicit cast (same
	# reasoning as lib/windows/com/__init__.py's own HRESULT constants:
	# these exceed i32's signed positive range as bit patterns).
	SEC_E_OK:                 SECURITY_STATUS = 0
	SEC_I_CONTINUE_NEEDED:    SECURITY_STATUS = 0x00090312
	SEC_I_CONTEXT_EXPIRED:    SECURITY_STATUS = 0x00090317
	SEC_E_INCOMPLETE_MESSAGE: SECURITY_STATUS = SECURITY_STATUS( 0x80090318 )
	SEC_E_WRONG_PRINCIPAL:    SECURITY_STATUS = SECURITY_STATUS( 0x80090322 )
	SEC_E_UNTRUSTED_ROOT:     SECURITY_STATUS = SECURITY_STATUS( 0x80090325 )
	SEC_E_ILLEGAL_MESSAGE:    SECURITY_STATUS = SECURITY_STATUS( 0x80090326 )
	SEC_E_CERT_EXPIRED:       SECURITY_STATUS = SECURITY_STATUS( 0x80090328 )
	SEC_E_BUFFER_TOO_SMALL:   SECURITY_STATUS = SECURITY_STATUS( 0x80090321 )
	SEC_E_ALGORITHM_MISMATCH: SECURITY_STATUS = SECURITY_STATUS( 0x80090331 )
	SEC_E_INVALID_TOKEN:      SECURITY_STATUS = SECURITY_STATUS( 0x80090308 )

elif compiler.target.os != 'macos':
	# Linux (and any other non-Windows, non-macOS target) - system OpenSSL.
	# Constants below (SSL_ERROR_*, X509_V_*, SSL_CTRL_SET_TLSEXT_HOSTNAME,
	# TLSEXT_NAMETYPE_host_name) are hardcoded from the real
	# openssl/ssl.h + openssl/x509_vfy.h (OpenSSL 3.5, Debian 13) - verified
	# against the actual headers via grep, not guessed, same bar as the
	# Windows constants above. `long` is 8 bytes on Linux x86_64 (LP64,
	# unlike Windows' LLP64 where `long` stays 4 bytes) - SSL_get_verify_
	# result's and SSL_ctrl's `long` parameters/returns are i64 here, not i32.

	SSL_VERIFY_PEER: i32 = 0x01
	SSL_VERIFY_NONE: i32 = 0x00

	SSL_ERROR_NONE:        i32 = 0
	SSL_ERROR_SSL:         i32 = 1
	SSL_ERROR_WANT_READ:   i32 = 2
	SSL_ERROR_WANT_WRITE:  i32 = 3
	SSL_ERROR_SYSCALL:     i32 = 5
	SSL_ERROR_ZERO_RETURN: i32 = 6

	# SSL_set_tlsext_host_name(s, name) is a macro in real openssl headers,
	# not an exported symbol - it expands to
	# SSL_ctrl(s, SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, name)
	# (confirmed against openssl/tls1.h) - so this file calls SSL_ctrl
	# directly with these two constants rather than @extern'ing a symbol
	# that doesn't exist.
	SSL_CTRL_SET_TLSEXT_HOSTNAME: i32 = 55
	TLSEXT_NAMETYPE_host_name:    i64 = 0

	X509_V_OK:                     i64 = 0
	X509_V_ERR_CERT_HAS_EXPIRED:   i64 = 10
	X509_V_ERR_HOSTNAME_MISMATCH:  i64 = 62


# SecHandle - real C typedefs CredHandle/CtxtHandle to this exact same
# struct (confirmed: sizeof == 16 for all three). dwLower/dwUpper are
# ULONG_PTR (pointer-sized) - usize matches on every target this compiles for.
@compiler.target( os = 'windows' )
@cstruct
class SecHandle:
	dwLower: usize = 0
	dwUpper: usize = 0

@compiler.target( os = 'windows' )
@cstruct
class SecBuffer:
	cbBuffer:   u32 = 0
	BufferType: u32 = 0
	pvBuffer:   Ptr[u8] = None

@compiler.target( os = 'windows' )
@cstruct
class SecBufferDesc:
	ulVersion: u32 = 0
	cBuffers:  u32 = 0
	pBuffers:  Ptr[SecBuffer] = None

# Four SecBuffer slots, laid out as four contiguous SecBuffer-shaped fields
# rather than a sys.alloc[SecBuffer](4) array - SecBuffer has no padding
# between instances (16 bytes, already 8-aligned), so &this cast to
# Ptr[SecBuffer] is byte-identical to a real C SecBuffer[4]. Sidesteps
# needing a generic sys.alloc[T](n) for a @cstruct element type (untested
# territory - every existing sys.alloc[T](n) call in this codebase is for
# T=u8) while staying within the proven "bare local + addrof" pattern
# lib/socket.py's own struct-construction code already established.
# InitializeSecurityContext only ever needs <=2, EncryptMessage/
# DecryptMessage need exactly 4 - this one shape covers every call site;
# cBuffers on the enclosing SecBufferDesc tells Schannel how many are live.
@compiler.target( os = 'windows' )
@cstruct
class _SecBuffer4:
	b0: SecBuffer = SecBuffer()
	b1: SecBuffer = SecBuffer()
	b2: SecBuffer = SecBuffer()
	b3: SecBuffer = SecBuffer()


# SCHANNEL_CRED - deprecated in favor of SCH_CREDENTIALS but still fully
# supported and much simpler (no nested TLS_PARAMETERS array); sufficient
# for a certificate-less TLS client. _pad0/_pad1 are real compiler-inserted
# padding on the x64 ABI (confirmed via offsetof - two DWORDs followed by a
# pointer needs 4 bytes of padding to reach 8-byte alignment), matching this
# codebase's existing convention of making struct padding explicit rather
# than relying on @cstruct to infer it (see lib/windows/kernel32.py's
# DynamicTimeZoneInformation _pad field).
@compiler.target( os = 'windows' )
@cstruct
class SCHANNEL_CRED:
	dwVersion:               u32 = 0
	cCreds:                  u32 = 0
	paCred:                  Ptr[None] = None
	hRootStore:              Ptr[None] = None
	cMappers:                u32 = 0
	_pad0:                   u32 = 0
	aphMappers:              Ptr[None] = None
	cSupportedAlgs:          u32 = 0
	_pad1:                   u32 = 0
	palgSupportedAlgs:       Ptr[None] = None
	grbitEnabledProtocols:   u32 = 0
	dwMinimumCipherStrength: u32 = 0
	dwMaximumCipherStrength: u32 = 0
	dwSessionLifespan:       u32 = 0
	dwFlags:                 u32 = 0
	dwCredFormat:            u32 = 0

@compiler.target( os = 'windows' )
@cstruct
class SecPkgContext_StreamSizes:
	cbHeader:         u32 = 0
	cbTrailer:        u32 = 0
	cbMaximumMessage: u32 = 0
	cBuffers:         u32 = 0
	cbBlockSize:      u32 = 0


@compiler.target( os = 'windows' )
def _sec_failed( status: SECURITY_STATUS ) -> bool:
	''' FAILED(HRESULT)-equivalent for SECURITY_STATUS - same convention
	(high bit set = failure), matching lib/windows/com/__init__.py's own
	FAILED(hr). '''
	return status < 0


# "A" (ANSI) entry points, not "W" - confirmed by the same real-header C
# probe/handshake that validated the struct layouts above: the ANSI SSPI
# entry points are fully supported for Schannel and avoid UTF-16 conversion
# entirely for pszTargetName/pszPackage, unlike e.g. CreateFileA/W where the
# two genuinely differ. Matches this file's use of str.get_cstr() throughout,
# the same idiom lib/socket.py already uses for inet_pton/getaddrinfo.

@compiler.target( os = 'windows' )
@extern( 'secur32', 'AcquireCredentialsHandleA' )
def AcquireCredentialsHandleA(
	pszPrincipal:    ConstPtr[u8],
	pszPackage:      ConstPtr[u8],
	fCredentialUse:  u32,
	pvLogonId:       Ptr[None],
	pAuthData:       Ptr[None],
	pGetKeyFn:       Ptr[None],
	pvGetKeyArgument: Ptr[None],
	phCredential:    Ptr[SecHandle],
	ptsExpiry:       Ptr[TimeStamp],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'InitializeSecurityContextA' )
def InitializeSecurityContextA(
	phCredential:   Ptr[SecHandle],
	phContext:      Ptr[SecHandle],
	pszTargetName:  ConstPtr[u8],
	fContextReq:    u32,
	Reserved1:      u32,
	TargetDataRep:  u32,
	pInput:         Ptr[SecBufferDesc],
	Reserved2:      u32,
	phNewContext:   Ptr[SecHandle],
	pOutput:        Ptr[SecBufferDesc],
	pfContextAttr:  Ptr[u32],
	ptsExpiry:      Ptr[TimeStamp],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'EncryptMessage' )
def EncryptMessage(
	phContext:    Ptr[SecHandle],
	fQOP:         u32,
	pMessage:     Ptr[SecBufferDesc],
	MessageSeqNo: u32,
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'DecryptMessage' )
def DecryptMessage(
	phContext:    Ptr[SecHandle],
	pMessage:     Ptr[SecBufferDesc],
	MessageSeqNo: u32,
	pfQOP:        Ptr[u32],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'QueryContextAttributesA' )
def QueryContextAttributesA(
	phContext:   Ptr[SecHandle],
	ulAttribute: u32,
	pBuffer:     Ptr[None],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'DeleteSecurityContext' )
def DeleteSecurityContext(
	phContext: Ptr[SecHandle],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'FreeCredentialsHandle' )
def FreeCredentialsHandle(
	phCredential: Ptr[SecHandle],
) -> SECURITY_STATUS:
	...

@compiler.target( os = 'windows' )
@extern( 'secur32', 'FreeContextBuffer' )
def FreeContextBuffer(
	pvContextBuffer: Ptr[None],
) -> SECURITY_STATUS:
	...


@compiler.target( os = 'windows' )
def _map_sec_error( status: SECURITY_STATUS ) -> SSLError:
	''' SEC_E_UNTRUSTED_ROOT/SEC_E_CERT_EXPIRED/SEC_E_WRONG_PRINCIPAL are
	confirmed-real, distinct, reachable codes (verified against
	self-signed.badssl.com / expired.badssl.com / wrong.host.badssl.com
	respectively with the standalone C client this was ported from) - not a
	guess at what Schannel might return. '''
	if status == SEC_E_UNTRUSTED_ROOT:
		return SSLError.CertificateVerifyFailed
	if status == SEC_E_CERT_EXPIRED:
		return SSLError.CertificateExpired
	if status == SEC_E_WRONG_PRINCIPAL:
		return SSLError.HostnameMismatch
	if status == SEC_E_ILLEGAL_MESSAGE or status == SEC_E_ALGORITHM_MISMATCH or status == SEC_E_INVALID_TOKEN or status == SEC_E_BUFFER_TOO_SMALL:
		return SSLError.ProtocolError
	return SSLError.HandshakeFailed


# ---------------------------------------------------------------------------
# _CipherBuf — growable raw-ciphertext accumulator read off the underlying
# socket, private to this file. Same grow/fill_from shape as lib/socket.py's
# own RecvBuffer, plus consume() (shift an already-processed prefix off the
# front) which the handshake/decrypt loops below need repeatedly. A private
# local copy rather than extending or reusing lib/socket.py's RecvBuffer -
# matches this codebase's own established precedent of each stdlib consumer
# keeping its own private growable buffer (see lib/http/client.py's
# _GrowableBuffer, per PLAN_HTTP_CLIENT.md) rather than sharing one, so this
# stays isolated from and doesn't risk lib/socket.py's already-shipped,
# tested code.
# ---------------------------------------------------------------------------

class _CipherBuf:
	__data: Ptr[u8]
	__len:  usize
	__cap:  usize

	def __init__( self, initial_cap: usize = 4096 ) -> None:
		self.__cap = initial_cap
		self.__data = sys.alloc[u8]( self.__cap )
		self.__len = 0

	def __del__( self ) -> None:
		sys.free( self.__data )

	def len( self ) -> usize:
		return self.__len

	def get_ptr( self ) -> Ptr[u8]:
		return self.__data

	def _grow( self, min_additional: usize ) -> None:
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			needed: usize = self.__len + min_additional
		if needed <= self.__cap:
			return
		new_cap: usize = self.__cap
		with compiler.panic_arithmetic( 'irrational buffer growth' ):
			while new_cap < needed:
				new_cap = new_cap * 2
		new_data: Ptr[u8] = sys.alloc[u8]( new_cap )
		sys.memcpy( new_data, self.__data, self.__len )
		sys.free( self.__data )
		self.__data = new_data
		self.__cap = new_cap

	def fill_from( self, sock: socket.Socket, chunk_size: usize = 4096 ) -> Result[usize, OSError]:
		''' one recv() call, appended to the buffer. Returns the number of
		bytes read - 0 means the peer closed the connection, same convention
		as Socket.recv()/RecvBuffer.fill_from() itself. '''
		self._grow( chunk_size )
		with compiler.wrap_arithmetic:
			dest: Ptr[u8] = self.__data + self.__len
			room: usize = self.__cap - self.__len
		n: usize = sock.recv( dest, room ).or_return()
		with compiler.wrap_arithmetic:
			self.__len += n
		return Result.Ok( n )

	def consume( self, n: usize ) -> None:
		''' discard n bytes from the front, shifting whatever's left down to
		offset 0. sys.memmove (not memcpy) - the source and destination
		regions genuinely overlap here (shifting the buffer's own tail down
		toward its head), unlike every other memcpy in this file, which
		always copies between two disjoint allocations. '''
		with compiler.panic_arithmetic( 'consume() cannot exceed buffered length' ):
			remaining: usize = self.__len - n
		if remaining > 0:
			with compiler.wrap_arithmetic:
				src: Ptr[u8] = self.__data + n
			sys.memmove( self.__data, src, remaining )
		self.__len = remaining


# ---------------------------------------------------------------------------
# SSLContext — holds the Schannel credential handle (roughly: "which
# protocols/validation policy"), independent of any one connection. One
# SSLContext can wrap_socket() many sockets.
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
class SSLContext:
	__cred: SecHandle

	def __del__( self ) -> None:
		cred: SecHandle = self.__cred
		FreeCredentialsHandle( compiler.addrof( cred ))

	@private
	def _cred( self ) -> SecHandle:
		return self.__cred

	@private
	@staticmethod
	def _from_cred( cred: SecHandle ) -> SSLContext:
		return SSLContext.__allocate__( __cred = cred )

	@staticmethod
	def create_default_context() -> Result[SSLContext, SSLError]:
		''' loads the Windows system trust store implicitly - Schannel
		validates the peer's certificate chain against it automatically
		during the handshake (SCH_CRED_AUTO_CRED_VALIDATION, the default
		validation behavior; SCH_CRED_NO_DEFAULT_CREDS means "don't send a
		client certificate", the normal case for an HTTPS client). '''
		pkg: str = "Microsoft Unified Security Protocol Provider"
		cred_data: SCHANNEL_CRED = SCHANNEL_CRED(
			dwVersion = SCHANNEL_CRED_VERSION,
			dwFlags = SCH_CRED_NO_DEFAULT_CREDS | SCH_CRED_AUTO_CRED_VALIDATION,
		)
		hcred: SecHandle = SecHandle()
		expiry: TimeStamp = TimeStamp( 0 )
		status: SECURITY_STATUS = AcquireCredentialsHandleA(
			None, pkg.get_cstr(), SECPKG_CRED_OUTBOUND, None,
			compiler.cast( Ptr[None], compiler.addrof( cred_data )),
			None, None,
			compiler.addrof( hcred ), compiler.addrof( expiry ),
		)
		if status != SEC_E_OK:
			return Result.Err( SSLError.Other )
		return Result.Ok( SSLContext._from_cred( hcred ))

	@staticmethod
	def create_unverified_context() -> Result[SSLContext, SSLError]:
		''' like create_default_context(), but SCH_CRED_MANUAL_CRED_VALIDATION
		instead of SCH_CRED_AUTO_CRED_VALIDATION - Schannel skips certificate
		chain validation entirely (expired/self-signed/hostname-mismatch all
		succeed) since nothing here ever calls the manual validation API a
		real "manual" caller would use to opt back in. Mirrors CPython's
		ssl._create_unverified_context() - for talking to a self-signed/dev
		server only, never for a real endpoint. '''
		pkg: str = "Microsoft Unified Security Protocol Provider"
		cred_data: SCHANNEL_CRED = SCHANNEL_CRED(
			dwVersion = SCHANNEL_CRED_VERSION,
			dwFlags = SCH_CRED_NO_DEFAULT_CREDS | SCH_CRED_MANUAL_CRED_VALIDATION,
		)
		hcred: SecHandle = SecHandle()
		expiry: TimeStamp = TimeStamp( 0 )
		status: SECURITY_STATUS = AcquireCredentialsHandleA(
			None, pkg.get_cstr(), SECPKG_CRED_OUTBOUND, None,
			compiler.cast( Ptr[None], compiler.addrof( cred_data )),
			None, None,
			compiler.addrof( hcred ), compiler.addrof( expiry ),
		)
		if status != SEC_E_OK:
			return Result.Err( SSLError.Other )
		return Result.Ok( SSLContext._from_cred( hcred ))


# ---------------------------------------------------------------------------
# SSLSocket — a Schannel security context bound to one connected
# socket.Socket, past the handshake. wrap_socket() ports a standalone C
# client 1:1 (see this file's header comment) - the InitializeSecurityContext
# loop shape (do_read toggling, SECBUFFER_EXTRA handling both mid-handshake
# and at completion) is deliberately unchanged from that validated reference,
# not simplified, since this exact shape is what real-world Schannel servers
# (some of which pipeline multiple handshake messages into one TCP segment)
# require.
# ---------------------------------------------------------------------------

@compiler.target( os = 'windows' )
class SSLSocket:
	__hctx:         SecHandle
	__sock:         socket.Socket
	__stream_sizes: SecPkgContext_StreamSizes
	__cipher:           _CipherBuf   # raw ciphertext not yet decrypted
	__plain:        Ptr[u8]      # decrypted plaintext not yet delivered to a caller
	__plain_len:    usize
	__plain_cap:    usize
	__plain_pos:    usize        # read cursor into __plain[0:__plain_len)
	__closed:       bool

	def __del__( self ) -> None:
		self.close()
		sys.free( self.__plain )

	def close( self ) -> None:
		if not self.__closed:
			hctx: SecHandle = self.__hctx
			DeleteSecurityContext( compiler.addrof( hctx ))
			self.__hctx = hctx
			self.__closed = True

	def _stash_plaintext( self, data: Ptr[u8], length: usize ) -> None:
		if length > self.__plain_cap:
			sys.free( self.__plain )
			self.__plain = sys.alloc[u8]( length )
			self.__plain_cap = length
		sys.memcpy( self.__plain, data, length )
		self.__plain_len = length
		self.__plain_pos = 0

	@staticmethod
	def wrap_socket( ctx: SSLContext, sock: socket.Socket, server_hostname: str ) -> Result[SSLSocket, SSLError]:
		hcred: SecHandle = ctx._cred()
		target: ConstPtr[u8] = server_hostname.get_cstr()
		flags: u32 = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT | ISC_REQ_CONFIDENTIALITY | ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_STREAM

		out_bufs0: _SecBuffer4 = _SecBuffer4( b0 = SecBuffer( BufferType = SECBUFFER_TOKEN ), b1 = SecBuffer(), b2 = SecBuffer(), b3 = SecBuffer() )
		out_desc0: SecBufferDesc = SecBufferDesc( ulVersion = SECBUFFER_VERSION, cBuffers = 1, pBuffers = compiler.cast( Ptr[SecBuffer], compiler.addrof( out_bufs0 )))

		hctx: SecHandle = SecHandle()
		expiry: TimeStamp = TimeStamp( 0 )
		out_flags0: u32 = 0
		status: SECURITY_STATUS = InitializeSecurityContextA(
			compiler.addrof( hcred ), None, target, flags, 0, SECURITY_NATIVE_DREP,
			None, 0, compiler.addrof( hctx ), compiler.addrof( out_desc0 ), compiler.addrof( out_flags0 ), compiler.addrof( expiry ),
		)
		if status != SEC_I_CONTINUE_NEEDED:
			return Result.Err( _map_sec_error( status ))

		if out_bufs0.b0.cbBuffer != 0 and out_bufs0.b0.pvBuffer is not None:
			with compiler.wrap_arithmetic:
				first_len: usize = usize( out_bufs0.b0.cbBuffer )
			match sock.send_all( compiler.cast( ConstPtr[u8], out_bufs0.b0.pvBuffer ), first_len ):
				case Result.Err( _ ):
					FreeContextBuffer( compiler.cast( Ptr[None], out_bufs0.b0.pvBuffer ))
					return Result.Err( SSLError.Other )
				case Result.Ok( _ ):
					pass
			FreeContextBuffer( compiler.cast( Ptr[None], out_bufs0.b0.pvBuffer ))

		in_buf: _CipherBuf = _CipherBuf()
		do_read: bool = True
		rounds: i32 = 0

		while status == SEC_I_CONTINUE_NEEDED or status == SEC_E_INCOMPLETE_MESSAGE:
			with compiler.wrap_arithmetic:
				rounds += 1
			if rounds > 50:
				return Result.Err( SSLError.HandshakeFailed )

			if do_read:
				match in_buf.fill_from( sock ):
					case Result.Ok( n ):
						if n == 0:
							return Result.Err( SSLError.Closed )
					case Result.Err( _ ):
						return Result.Err( SSLError.Other )

			with compiler.saturate_arithmetic:
				in_len_u32: u32 = u32( in_buf.len() )

			in_bufs: _SecBuffer4 = _SecBuffer4(
				b0 = SecBuffer( cbBuffer = in_len_u32, BufferType = SECBUFFER_TOKEN, pvBuffer = in_buf.get_ptr() ),
				b1 = SecBuffer( cbBuffer = 0, BufferType = SECBUFFER_EMPTY, pvBuffer = None ),
				b2 = SecBuffer(), b3 = SecBuffer(),
			)
			in_desc: SecBufferDesc = SecBufferDesc( ulVersion = SECBUFFER_VERSION, cBuffers = 2, pBuffers = compiler.cast( Ptr[SecBuffer], compiler.addrof( in_bufs )))

			out_bufs: _SecBuffer4 = _SecBuffer4( b0 = SecBuffer( BufferType = SECBUFFER_TOKEN ), b1 = SecBuffer(), b2 = SecBuffer(), b3 = SecBuffer() )
			out_desc: SecBufferDesc = SecBufferDesc( ulVersion = SECBUFFER_VERSION, cBuffers = 1, pBuffers = compiler.cast( Ptr[SecBuffer], compiler.addrof( out_bufs )))

			out_flags: u32 = 0
			status = InitializeSecurityContextA(
				compiler.addrof( hcred ), compiler.addrof( hctx ), None, flags, 0, SECURITY_NATIVE_DREP,
				compiler.addrof( in_desc ), 0, None, compiler.addrof( out_desc ), compiler.addrof( out_flags ), compiler.addrof( expiry ),
			)

			if status == SEC_E_OK or status == SEC_I_CONTINUE_NEEDED or ( _sec_failed( status ) and ( out_flags & ISC_RET_EXTRA_DATA ) != 0 ):
				if out_bufs.b0.cbBuffer != 0 and out_bufs.b0.pvBuffer is not None:
					with compiler.wrap_arithmetic:
						out_len: usize = usize( out_bufs.b0.cbBuffer )
					sock.send_all( compiler.cast( ConstPtr[u8], out_bufs.b0.pvBuffer ), out_len ).is_ok()
					FreeContextBuffer( compiler.cast( Ptr[None], out_bufs.b0.pvBuffer ))

			if status == SEC_E_INCOMPLETE_MESSAGE:
				do_read = True
				continue

			if status == SEC_E_OK:
				if in_bufs.b1.BufferType == SECBUFFER_EXTRA:
					with compiler.wrap_arithmetic:
						extra_len: usize = usize( in_bufs.b1.cbBuffer )
						consumed: usize = in_buf.len() - extra_len
					in_buf.consume( consumed )
				else:
					in_buf.consume( in_buf.len() )
				break

			if _sec_failed( status ):
				return Result.Err( _map_sec_error( status ))

			# still SEC_I_CONTINUE_NEEDED - a pipelined server may have sent
			# more than one handshake message in the same TCP segment; if
			# Schannel only consumed a prefix, reprocess the remainder
			# immediately instead of blocking on a read that may never come.
			if in_bufs.b1.BufferType == SECBUFFER_EXTRA:
				with compiler.wrap_arithmetic:
					extra_len2: usize = usize( in_bufs.b1.cbBuffer )
					consumed2: usize = in_buf.len() - extra_len2
				in_buf.consume( consumed2 )
				do_read = False
			else:
				in_buf.consume( in_buf.len() )
				do_read = True

		sizes: SecPkgContext_StreamSizes = SecPkgContext_StreamSizes()
		QueryContextAttributesA( compiler.addrof( hctx ), SECPKG_ATTR_STREAM_SIZES, compiler.cast( Ptr[None], compiler.addrof( sizes )))

		return Result.Ok( SSLSocket.__allocate__(
			__hctx = hctx,
			__sock = sock,
			__stream_sizes = sizes,
			__cipher = in_buf,
			__plain = sys.alloc[u8]( 4096 ),
			__plain_len = 0,
			__plain_cap = usize( 4096 ),
			__plain_pos = 0,
			__closed = False,
		))

	def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, SSLError]:
		with compiler.saturate_arithmetic:
			max_msg: usize = usize( self.__stream_sizes.cbMaximumMessage )
		send_len: usize = count if count < max_msg else max_msg

		with compiler.wrap_arithmetic:
			header_off: usize = usize( self.__stream_sizes.cbHeader )
			trailer_off: usize = header_off + send_len
			total: usize = trailer_off + usize( self.__stream_sizes.cbTrailer )
		msg_buf: Ptr[u8] = sys.alloc[u8]( total )
		with compiler.wrap_arithmetic:
			data_ptr: Ptr[u8] = msg_buf + header_off
			trailer_ptr: Ptr[u8] = msg_buf + trailer_off
		sys.memcpy( data_ptr, buf, send_len )

		with compiler.saturate_arithmetic:
			send_len_u32: u32 = u32( send_len )

		bufs: _SecBuffer4 = _SecBuffer4(
			b0 = SecBuffer( cbBuffer = self.__stream_sizes.cbHeader, BufferType = SECBUFFER_STREAM_HEADER, pvBuffer = msg_buf ),
			b1 = SecBuffer( cbBuffer = send_len_u32, BufferType = SECBUFFER_DATA, pvBuffer = data_ptr ),
			b2 = SecBuffer( cbBuffer = self.__stream_sizes.cbTrailer, BufferType = SECBUFFER_STREAM_TRAILER, pvBuffer = trailer_ptr ),
			b3 = SecBuffer(),
		)
		desc: SecBufferDesc = SecBufferDesc( ulVersion = SECBUFFER_VERSION, cBuffers = 4, pBuffers = compiler.cast( Ptr[SecBuffer], compiler.addrof( bufs )))

		hctx: SecHandle = self.__hctx
		status: SECURITY_STATUS = EncryptMessage( compiler.addrof( hctx ), 0, compiler.addrof( desc ), 0 )
		self.__hctx = hctx
		if _sec_failed( status ):
			sys.free( msg_buf )
			return Result.Err( _map_sec_error( status ))

		with compiler.wrap_arithmetic:
			total_send: usize = usize( bufs.b0.cbBuffer ) + usize( bufs.b1.cbBuffer ) + usize( bufs.b2.cbBuffer )

		result = self.__sock.send_all( compiler.cast( ConstPtr[u8], msg_buf ), total_send )
		sys.free( msg_buf )
		match result:
			case Result.Ok( _ ):
				return Result.Ok( send_len )
			case Result.Err( _ ):
				return Result.Err( SSLError.Other )

	def send_all( self, buf: ConstPtr[u8], count: usize ) -> Result[None, SSLError]:
		sent: usize = 0
		with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
			while sent < count:
				n: usize = self.send( buf + sent, count - sent ).or_return()
				if n == 0:
					return Result.Err( SSLError.Closed )
				sent += n
		return Result.Ok( None )

	def recv( self, buf: Ptr[u8], count: usize ) -> Result[usize, SSLError]:
		if self.__plain_pos < self.__plain_len:
			with compiler.wrap_arithmetic:
				avail: usize = self.__plain_len - self.__plain_pos
			take: usize = count if count < avail else avail
			with compiler.wrap_arithmetic:
				src: Ptr[u8] = self.__plain + self.__plain_pos
			sys.memcpy( buf, src, take )
			with compiler.wrap_arithmetic:
				self.__plain_pos += take
			return Result.Ok( take )

		while True:
			if self.__cipher.len() == 0:
				match self.__cipher.fill_from( self.__sock ):
					case Result.Ok( n ):
						if n == 0:
							return Result.Ok( 0 )
					case Result.Err( _ ):
						return Result.Err( SSLError.Other )

			with compiler.saturate_arithmetic:
				in_len_u32: u32 = u32( self.__cipher.len() )

			dec_bufs: _SecBuffer4 = _SecBuffer4(
				b0 = SecBuffer( cbBuffer = in_len_u32, BufferType = SECBUFFER_DATA, pvBuffer = self.__cipher.get_ptr() ),
				b1 = SecBuffer(), b2 = SecBuffer(), b3 = SecBuffer(),
			)
			dec_desc: SecBufferDesc = SecBufferDesc( ulVersion = SECBUFFER_VERSION, cBuffers = 4, pBuffers = compiler.cast( Ptr[SecBuffer], compiler.addrof( dec_bufs )))

			hctx: SecHandle = self.__hctx
			status: SECURITY_STATUS = DecryptMessage( compiler.addrof( hctx ), compiler.addrof( dec_desc ), 0, None )
			self.__hctx = hctx

			if status == SEC_E_INCOMPLETE_MESSAGE:
				match self.__cipher.fill_from( self.__sock ):
					case Result.Ok( n ):
						if n == 0:
							return Result.Err( SSLError.ProtocolError )
					case Result.Err( _ ):
						return Result.Err( SSLError.Other )
				continue

			if status == SEC_I_CONTEXT_EXPIRED:
				return Result.Ok( 0 )

			if _sec_failed( status ):
				return Result.Err( _map_sec_error( status ))

			# SEC_E_OK - exactly one of b1/b2/b3 becomes SECBUFFER_DATA (the
			# plaintext), at most one becomes SECBUFFER_EXTRA (leftover
			# ciphertext for the next record) - mirrors the standalone C
			# reference's own buffers[1..3] walk.
			data_ptr: Ptr[u8] = None
			data_len: usize = 0
			extra_len: usize = 0
			found_extra: bool = False

			if dec_bufs.b1.BufferType == SECBUFFER_DATA:
				data_ptr = dec_bufs.b1.pvBuffer
				data_len = usize( dec_bufs.b1.cbBuffer )
			elif dec_bufs.b1.BufferType == SECBUFFER_EXTRA:
				extra_len = usize( dec_bufs.b1.cbBuffer )
				found_extra = True

			if dec_bufs.b2.BufferType == SECBUFFER_DATA:
				data_ptr = dec_bufs.b2.pvBuffer
				data_len = usize( dec_bufs.b2.cbBuffer )
			elif dec_bufs.b2.BufferType == SECBUFFER_EXTRA:
				extra_len = usize( dec_bufs.b2.cbBuffer )
				found_extra = True

			if dec_bufs.b3.BufferType == SECBUFFER_DATA:
				data_ptr = dec_bufs.b3.pvBuffer
				data_len = usize( dec_bufs.b3.cbBuffer )
			elif dec_bufs.b3.BufferType == SECBUFFER_EXTRA:
				extra_len = usize( dec_bufs.b3.cbBuffer )
				found_extra = True

			# stash the plaintext (which points INTO self.__cipher's own memory)
			# before consume() shifts/overwrites that same memory
			if data_len > 0:
				self._stash_plaintext( data_ptr, data_len )

			if found_extra:
				with compiler.wrap_arithmetic:
					consumed: usize = self.__cipher.len() - extra_len
				self.__cipher.consume( consumed )
			else:
				self.__cipher.consume( self.__cipher.len() )

			if data_len == 0:
				continue  # e.g. a bare alert/renegotiation record - decrypt the next one
			break

		with compiler.wrap_arithmetic:
			avail2: usize = self.__plain_len - self.__plain_pos
		take2: usize = count if count < avail2 else avail2
		with compiler.wrap_arithmetic:
			src2: Ptr[u8] = self.__plain + self.__plain_pos
		sys.memcpy( buf, src2, take2 )
		with compiler.wrap_arithmetic:
			self.__plain_pos += take2
		return Result.Ok( take2 )


# ---------------------------------------------------------------------------
# Linux backend — system OpenSSL (libssl, via @compiler.target(has_library=
# (...))). Every extern signature below was cross-checked against the real
# openssl/ssl.h (OpenSSL 3.5, Debian 13 trixie) rather than hand-derived from
# memory, and the full connect/verify/read/write sequence was validated
# end-to-end with a standalone C client (real TCP + real OpenSSL handshake
# against example.com, and the same three badssl.com cert-failure paths the
# Windows backend was checked against) before being transcribed here - same
# bar as the Windows backend above. Unlike Schannel, every OpenSSL type used
# here (SSL_CTX*, SSL*, SSL_METHOD*) is fully opaque from this side - we
# never allocate or read a single field of any of them, only pass the
# pointers back to libssl - so there are no struct layouts to get wrong here,
# just function signatures and integer constants.
#
# @compiler.target(os=..., has_library=...) - a single decorator combining
# both conditions - is deliberately used for every def/class below (matching
# lib/crt.py's own decorator-per-extern style) rather than nesting a second
# `if compiler.has_library(...)` inside the module-level `elif` above that
# already holds this backend's constants: only TypeAlias/const bindings are
# confirmed to work inside a bare module-level if/else (see this file's
# Windows section comment); nesting a second conditional block inside that
# one, specifically to hold @extern defs, is untested territory this file
# doesn't need to risk when the decorator form already proves both
# conditions compose (discovery.py's _matches_active_target checks every
# keyword given, has_library included).
# ---------------------------------------------------------------------------

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'TLS_client_method' )
def TLS_client_method() -> Ptr[None]:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_CTX_new' )
def SSL_CTX_new( method: Ptr[None] ) -> Ptr[None]:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_CTX_free' )
def SSL_CTX_free( ctx: Ptr[None] ) -> None:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_CTX_set_verify' )
def SSL_CTX_set_verify( ctx: Ptr[None], mode: i32, callback: Ptr[None] ) -> None:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_CTX_set_default_verify_paths' )
def SSL_CTX_set_default_verify_paths( ctx: Ptr[None] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_new' )
def SSL_new( ctx: Ptr[None] ) -> Ptr[None]:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_free' )
def SSL_free( ssl: Ptr[None] ) -> None:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_set_fd' )
def SSL_set_fd( ssl: Ptr[None], fd: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_set1_host' )
def SSL_set1_host( ssl: Ptr[None], host: ConstPtr[u8] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_ctrl' )
def SSL_ctrl( ssl: Ptr[None], cmd: i32, larg: i64, parg: Ptr[None] ) -> i64:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_connect' )
def SSL_connect( ssl: Ptr[None] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_read' )
def SSL_read( ssl: Ptr[None], buf: Ptr[None], num: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_write' )
def SSL_write( ssl: Ptr[None], buf: Ptr[None], num: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_get_error' )
def SSL_get_error( ssl: Ptr[None], ret_code: i32 ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_shutdown' )
def SSL_shutdown( ssl: Ptr[None] ) -> i32:
	...

@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
@extern( 'ssl', 'SSL_get_verify_result' )
def SSL_get_verify_result( ssl: Ptr[None] ) -> i64:
	...


@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
def _map_ssl_error( ssl: Ptr[None], err: i32 ) -> SSLError:
	''' SSL_get_error() alone only distinguishes SSL_ERROR_SSL (a generic
	"the library/protocol rejected something") from the transport-level
	codes - the specific certificate-failure reason lives in a SEPARATE call,
	SSL_get_verify_result(), which is only meaningful once SSL_ERROR_SSL is
	confirmed (real OpenSSL behavior, not an assumption - confirmed by the
	same standalone C client this was ported from: SSL_get_error() returned
	1 (SSL_ERROR_SSL) uniformly for all three badssl.com failures, and only
	SSL_get_verify_result() distinguished expired (10) / wrong-host (62) /
	self-signed (an X509_V_ERR_* other than those two, hence the trailing
	catch-all) from each other. '''
	if err == SSL_ERROR_ZERO_RETURN:
		return SSLError.Closed
	if err == SSL_ERROR_SYSCALL:
		return SSLError.Other
	if err != SSL_ERROR_SSL:
		return SSLError.HandshakeFailed
	vr: i64 = SSL_get_verify_result( ssl )
	if vr == X509_V_ERR_CERT_HAS_EXPIRED:
		return SSLError.CertificateExpired
	if vr == X509_V_ERR_HOSTNAME_MISMATCH:
		return SSLError.HostnameMismatch
	if vr != X509_V_OK:
		return SSLError.CertificateVerifyFailed
	return SSLError.HandshakeFailed


@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
class SSLContext:
	__ctx: Ptr[None]  # SSL_CTX*

	def __del__( self ) -> None:
		SSL_CTX_free( self.__ctx )

	@private
	def _ctx( self ) -> Ptr[None]:
		return self.__ctx

	@private
	@staticmethod
	def _from_ctx( ctx: Ptr[None] ) -> SSLContext:
		return SSLContext.__allocate__( __ctx = ctx )

	@staticmethod
	def create_default_context() -> Result[SSLContext, SSLError]:
		''' loads the system trust store implicitly
		(SSL_CTX_set_default_verify_paths - OpenSSL's own default CA bundle/
		directory search, same spirit as Schannel's automatic system-store
		validation on Windows) and turns on peer certificate verification
		(SSL_CTX_set_verify(..., SSL_VERIFY_PEER, ...) - OFF by default in
		raw OpenSSL, a well-known footgun this wrapper doesn't expose). '''
		method: Ptr[None] = TLS_client_method()
		ctx: Ptr[None] = SSL_CTX_new( method )
		if ctx is None:
			return Result.Err( SSLError.Other )
		SSL_CTX_set_verify( ctx, SSL_VERIFY_PEER, None )
		if SSL_CTX_set_default_verify_paths( ctx ) != 1:
			SSL_CTX_free( ctx )
			return Result.Err( SSLError.Other )
		return Result.Ok( SSLContext._from_ctx( ctx ))

	@staticmethod
	def create_unverified_context() -> Result[SSLContext, SSLError]:
		''' like create_default_context(), but SSL_VERIFY_NONE instead of
		SSL_VERIFY_PEER - the handshake succeeds regardless of the peer's
		certificate (expired/self-signed/hostname-mismatch all succeed).
		Skips loading the system trust store too, since nothing here needs
		it with verification off. Mirrors CPython's
		ssl._create_unverified_context() - for talking to a self-signed/dev
		server only, never for a real endpoint. '''
		method: Ptr[None] = TLS_client_method()
		ctx: Ptr[None] = SSL_CTX_new( method )
		if ctx is None:
			return Result.Err( SSLError.Other )
		SSL_CTX_set_verify( ctx, SSL_VERIFY_NONE, None )
		return Result.Ok( SSLContext._from_ctx( ctx ))


@compiler.target( os = not ( 'windows', 'macos' ), has_library = ( 'ssl', 'SSL_new' ) )
class SSLSocket:
	__ssl:    Ptr[None]  # SSL*
	__sock:   socket.Socket
	__closed: bool

	def __del__( self ) -> None:
		self.close()

	def close( self ) -> None:
		if not self.__closed:
			SSL_shutdown( self.__ssl )
			SSL_free( self.__ssl )
			self.__closed = True

	@staticmethod
	def wrap_socket( ctx: SSLContext, sock: socket.Socket, server_hostname: str ) -> Result[SSLSocket, SSLError]:
		ssl: Ptr[None] = SSL_new( ctx._ctx() )
		if ssl is None:
			return Result.Err( SSLError.Other )

		fd: i32 = sock.fileno()
		if SSL_set_fd( ssl, fd ) != 1:
			SSL_free( ssl )
			return Result.Err( SSLError.Other )

		host_cstr: ConstPtr[u8] = server_hostname.get_cstr()
		# SNI - see this section's own header comment on why this calls
		# SSL_ctrl directly instead of a nonexistent SSL_set_tlsext_host_name
		# symbol. Return value intentionally unchecked, matching the
		# validated C reference (a failed SNI call still lets the handshake
		# proceed - it just means the server can't select a cert by name).
		SSL_ctrl( ssl, SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name, compiler.cast( Ptr[None], host_cstr ))
		if SSL_set1_host( ssl, host_cstr ) != 1:
			SSL_free( ssl )
			return Result.Err( SSLError.Other )

		rc: i32 = SSL_connect( ssl )
		if rc != 1:
			err: i32 = SSL_get_error( ssl, rc )
			mapped: SSLError = _map_ssl_error( ssl, err )
			SSL_free( ssl )
			return Result.Err( mapped )

		return Result.Ok( SSLSocket.__allocate__( __ssl = ssl, __sock = sock, __closed = False ))

	def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, SSLError]:
		with compiler.saturate_arithmetic:
			n: i32 = i32( count )
		rc: i32 = SSL_write( self.__ssl, compiler.cast( Ptr[None], buf ), n )
		if rc <= 0:
			err: i32 = SSL_get_error( self.__ssl, rc )
			return Result.Err( _map_ssl_error( self.__ssl, err ))
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( rc ))

	def send_all( self, buf: ConstPtr[u8], count: usize ) -> Result[None, SSLError]:
		sent: usize = 0
		with compiler.panic_arithmetic( 'bounded by count, cannot overflow' ):
			while sent < count:
				n: usize = self.send( buf + sent, count - sent ).or_return()
				if n == 0:
					return Result.Err( SSLError.Closed )
				sent += n
		return Result.Ok( None )

	def recv( self, buf: Ptr[u8], count: usize ) -> Result[usize, SSLError]:
		with compiler.saturate_arithmetic:
			n: i32 = i32( count )
		rc: i32 = SSL_read( self.__ssl, compiler.cast( Ptr[None], buf ), n )
		if rc <= 0:
			err: i32 = SSL_get_error( self.__ssl, rc )
			if err == SSL_ERROR_ZERO_RETURN:
				return Result.Ok( 0 )
			# a clean EOF without a close_notify alert (SSL_ERROR_SYSCALL
			# with rc == 0) is common in practice - consistent with
			# Socket.recv()'s own "0 means peer closed" convention, treated
			# as a normal close rather than a hard error.
			if err == SSL_ERROR_SYSCALL and rc == 0:
				return Result.Ok( 0 )
			return Result.Err( _map_ssl_error( self.__ssl, err ))
		with compiler.wrap_arithmetic:
			return Result.Ok( usize( rc ))


# ---------------------------------------------------------------------------
# macOS — deliberately NOT implemented (see PLAN_SSL.md: Secure Transport /
# Network.framework, blocked on having a real Mac to verify a handshake
# against - "do the binding/struct work but hold off calling it done without
# a real handshake test" was the standing rule for Windows/Linux too, and
# there's no way to honor that here yet).
#
# This is a real class (not just an absent one) for a concrete reason found
# by testing this exact scenario against a simulated macos compiler target
# (Discovery(active_target={'os': 'macos', ...})): lib/http/client.py's
# _Transport @union declares `Secure: ssl.SSLSocket` as a field type
# UNCONDITIONALLY (a union needs one concrete type per variant, not a
# per-target one) - if ssl.SSLSocket didn't exist at all on macOS,
# type_resolver.py's RC-class destructor synthesis crashes outright
# (AttributeError: 'NoneType' object has no attribute 'is_rc_pointer', deep
# in _build_field_teardown_ast) the moment ANY program merely imports
# lib/http/client.py on a macOS target - even one that only ever uses
# plain http://, never touches TLS. So SSLContext/SSLSocket need to exist
# as real, structurally valid types on every target lib/http/client.py
# might compile for, whether or not that target's TLS backend is finished.
#
# Given that, every method that would need to actually DO something routes
# through _MACOS_SSL_NOT_YET_IMPLEMENTED - a deliberately undefined name, not
# a typo. MetalPy has no compiler.error(...)/compiler.static_assert(...)
# intrinsic to raise a custom compile-time message (confirmed absent from
# discovery.py/compile_time_transformer.py), so this is the mechanism that
# exists: referencing an undefined name inside a function body only gets
# type-checked once something actually reaches/calls that function (stage 2
# - "walk the tree from main and determine everything touched by main" per
# ARCHITECTURE.md). A program that merely `import ssl` and never touches TLS
# compiles clean. lib/http/client.py itself is a different story, NOT because
# of anything specific to this file: its _transport_send/_transport_recv/
# _transport_close each pattern-match both _Transport variants in one shared
# function body, and a function's whole body - every match arm, not just the
# one actually taken - is what gets compiled, so those three functions reach
# SSLSocket.send/recv/close whenever THEY are reached, regardless of whether
# a given call is plain http:// or https:// - meaning any http.client usage
# at all currently hits this pill on macOS, same as it already hit the
# pre-poison-pill crash described above for the identical structural reason
# (see PLAN_SSL.md's own "macOS poison pill" section for the full trade-off).
# Either way, compilation fails with a loud, self-explanatory error instead
# of silently miscompiling or crashing the compiler itself:
#   name '_MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD' is not defined
# ---------------------------------------------------------------------------

@compiler.target( os = 'macos' )
class SSLContext:
	@staticmethod
	def create_default_context() -> Result[SSLContext, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

	@staticmethod
	def create_unverified_context() -> Result[SSLContext, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

@compiler.target( os = 'macos' )
class SSLSocket:
	# a real field (not zero fields) so this stays a structurally ordinary
	# RC class matching the Windows/Linux backends' own shape - never
	# actually populated, since wrap_socket() below never returns Ok(...).
	__unused: bool

	@staticmethod
	def wrap_socket( ctx: SSLContext, sock: socket.Socket, server_hostname: str ) -> Result[SSLSocket, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

	def send( self, buf: ConstPtr[u8], count: usize ) -> Result[usize, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

	def send_all( self, buf: ConstPtr[u8], count: usize ) -> Result[None, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

	def recv( self, buf: Ptr[u8], count: usize ) -> Result[usize, SSLError]:
		return _MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()

	def close( self ) -> None:
		_MACOS_SSL_NOT_YET_IMPLEMENTED__SEE_PLAN_SSL_MD()
