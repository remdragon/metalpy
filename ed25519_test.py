import unittest

import test_support
from compiler import Compiler
from discovery import Discovery


class Ed25519Tests( test_support.RealCompileMixin, unittest.TestCase ):
	''' Real compile-and-run coverage for lib/ed25519.py, against the
	official RFC 8032 SS7.1 Ed25519 test vectors (TEST 1: empty message,
	TEST 2: 1-byte message) plus this module's own generate/sign/verify
	round trip. '''

	def setUp( self ) -> None:
		self.discovery = Discovery( import_builtins = True )
		self.compiler = Compiler( self.discovery )

	@unittest.skipUnless( test_support.HAS_CC, 'no C compiler (clang/gcc/msvc) found - skipping' )
	def test_programs_compile_and_run( self ) -> None:
		self.assert_programs_run([
			# RFC 8032 SS7.1 TEST 1 - empty message.
			( 'rfc8032_test1_empty_message', '''
import base64
import ed25519

def bytes_eq( a: bytes, b: bytes ) -> bool:
	if len( a ) != len( b ):
		return False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( a ):
			if a.__getitem__( i ).unwrap( 'a[i]' ) != b.__getitem__( i ).unwrap( 'b[i]' ):
				return False
			i += usize( 1 )
	return True

def main() -> i32:
	sk: bytes = base64.b16decode( '9D61B19DEFFD5A60BA844AF492EC2CC44449C5697B326919703BAC031CAE7F60'.encode().unwrap( 'x' )).unwrap( 'sk' )
	expected_pk: bytes = base64.b16decode( 'D75A980182B10AB7D54BFED3C964073A0EE172F3DAA62325AF021A68F707511A'.encode().unwrap( 'x' )).unwrap( 'pk' )
	expected_sig: bytes = base64.b16decode( 'E5564300C360AC729086E2CC806E828A84877F1EB8E5D974D873E065224901555FB8821590A33BACC61E39701CF9B46BD25BF5F0595BBE24655141438E7A100B'.encode().unwrap( 'x' )).unwrap( 'sig' )
	empty: bytes = bytes.from_bytearray( move( bytearray( 0 )))

	pk: bytes = ed25519.public_key_from_seed( sk )
	if not bytes_eq( pk, expected_pk ):
		return 1

	sig: bytes = ed25519.sign( sk, empty ).unwrap( 'sign' )
	if not bytes_eq( sig[:usize(32)], expected_sig[:usize(32)] ):
		return 4
	if not bytes_eq( sig[usize(32):], expected_sig[usize(32):] ):
		return 5
	if not bytes_eq( sig, expected_sig ):
		return 2

	if not ed25519.verify( pk, empty, sig ):
		return 3
	return 0
''' ),
			# RFC 8032 SS7.1 TEST 2 - 1-byte message (0x72).
			( 'rfc8032_test2_one_byte_message', '''
import base64
import ed25519

def bytes_eq( a: bytes, b: bytes ) -> bool:
	if len( a ) != len( b ):
		return False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( a ):
			if a.__getitem__( i ).unwrap( 'a[i]' ) != b.__getitem__( i ).unwrap( 'b[i]' ):
				return False
			i += usize( 1 )
	return True

def main() -> i32:
	sk: bytes = base64.b16decode( '4CCD089B28FF96DA9DB6C346EC114E0F5B8A319F35ABA624DA8CF6ED4FB8A6FB'.encode().unwrap( 'x' )).unwrap( 'sk' )
	expected_pk: bytes = base64.b16decode( '3D4017C3E843895A92B70AA74D1B7EBC9C982CCF2EC4968CC0CD55F12AF4660C'.encode().unwrap( 'x' )).unwrap( 'pk' )
	expected_sig: bytes = base64.b16decode( '92A009A9F0D4CAB8720E820B5F642540A2B27B5416503F8FB3762223EBDB69DA085AC1E43E15996E458F3613D0F11D8C387B2EAEB4302AEEB00D291612BB0C00'.encode().unwrap( 'x' )).unwrap( 'sig' )
	msg: bytes = base64.b16decode( '72'.encode().unwrap( 'x' )).unwrap( 'msg' )

	pk: bytes = ed25519.public_key_from_seed( sk )
	if not bytes_eq( pk, expected_pk ):
		return 1

	sig: bytes = ed25519.sign( sk, msg ).unwrap( 'sign' )
	if not bytes_eq( sig, expected_sig ):
		return 2

	if not ed25519.verify( pk, msg, sig ):
		return 3
	return 0
''' ),
			( 'keypair_pubkey_and_privkey_are_32_bytes', '''
import ed25519

def main() -> i32:
	( pub, priv ) = ed25519.generate_keypair().unwrap( 'generate_keypair' )
	if len( pub ) != usize( 32 ):
		return 1
	if len( priv ) != usize( 32 ):
		return 2
	return 0
''' ),
			( 'sign_then_verify_round_trip', '''
import ed25519

def main() -> i32:
	( pub, priv ) = ed25519.generate_keypair().unwrap( 'generate_keypair' )
	msg: bytes = 'attack at dawn'.encode().unwrap( 'msg' )
	sig: bytes = ed25519.sign( priv, msg ).unwrap( 'sign' )
	if not ed25519.verify( pub, msg, sig ):
		return 1
	return 0
''' ),
			( 'tampered_message_rejected', '''
import ed25519

def main() -> i32:
	( pub, priv ) = ed25519.generate_keypair().unwrap( 'generate_keypair' )
	msg: bytes = 'attack at dawn'.encode().unwrap( 'msg' )
	tampered: bytes = 'attack at dusk'.encode().unwrap( 'tampered' )
	sig: bytes = ed25519.sign( priv, msg ).unwrap( 'sign' )
	if ed25519.verify( pub, tampered, sig ):
		return 1
	return 0
''' ),
			( 'tampered_signature_rejected', '''
import ed25519

def main() -> i32:
	( pub, priv ) = ed25519.generate_keypair().unwrap( 'generate_keypair' )
	msg: bytes = 'attack at dawn'.encode().unwrap( 'msg' )
	sig: bytes = ed25519.sign( priv, msg ).unwrap( 'sign' )
	bad: bytearray = bytearray( usize( 64 ))
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < usize( 64 ):
			bad[i] = sig.__getitem__( i ).unwrap( 'sig[i]' )
			i += usize( 1 )
		last: u8 = bad.__getitem__( usize( 63 )).unwrap( 'bad[63]' )
		bad[63] = last ^ u8( 1 )
	tampered_sig: bytes = bytes.from_bytearray( move( bad ))
	if ed25519.verify( pub, msg, tampered_sig ):
		return 1
	return 0
''' ),
			( 'wrong_public_key_rejected', '''
import ed25519

def main() -> i32:
	( pub_a, priv_a ) = ed25519.generate_keypair().unwrap( 'a' )
	( pub_b, priv_b ) = ed25519.generate_keypair().unwrap( 'b' )
	msg: bytes = 'attack at dawn'.encode().unwrap( 'msg' )
	sig: bytes = ed25519.sign( priv_a, msg ).unwrap( 'sign' )
	if ed25519.verify( pub_b, msg, sig ):
		return 1
	return 0
''' ),
			( 'independent_keypairs_differ', '''
import ed25519

def bytes_eq( a: bytes, b: bytes ) -> bool:
	if len( a ) != len( b ):
		return False
	i: usize = 0
	with compiler.wrap_arithmetic:
		while i < len( a ):
			if a.__getitem__( i ).unwrap( 'a[i]' ) != b.__getitem__( i ).unwrap( 'b[i]' ):
				return False
			i += usize( 1 )
	return True

def main() -> i32:
	( pub_a, priv_a ) = ed25519.generate_keypair().unwrap( 'a' )
	( pub_b, priv_b ) = ed25519.generate_keypair().unwrap( 'b' )
	if bytes_eq( pub_a, pub_b ):
		return 1
	if bytes_eq( priv_a, priv_b ):
		return 2
	return 0
''' ),
		] )


if __name__ == '__main__':
	unittest.main()
