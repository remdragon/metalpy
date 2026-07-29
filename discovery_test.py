# stdlib imports
import logging
from pathlib import Path
import unittest

# local imports
import discovery
from mpy_types import Module, NameRegistry

logger = logging.getLogger( __name__ )

class Tests( unittest.TestCase ):
	def setUp( self ) -> None:
		self.registry = NameRegistry()
		self.discovery = discovery.Discovery( self.registry )
	
	def test_imports( self ) -> None:
		test_self = self
		
		class TestComplete( BaseException ):
			pass
		
		class MockDiscovery( discovery.Discovery ):
			expect_package: str
			
			@staticmethod
			def new_test( expect_package: str ) -> MockDiscovery:
				disco = MockDiscovery( NameRegistry(), import_builtins = False )
				disco.expect_package = expect_package
				return disco
			
			def import_name( self, package: str ) -> discovery.Module:
				test_self.assertEqual( package, self.expect_package )
				raise TestComplete()
			
			def import_code( self, code: str, filename: Path, scope: str|None = None ) -> Module:
				try:
					super().import_code( code, filename, scope )
				except TestComplete:
					return None
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = None )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = 'codecs' )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'import codecs', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )
		
		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = None )
		
		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )
		
		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'import codecs.utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from codecs import utf8', Path( '__irrelevant__.py' ), scope = None )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from codecs import utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from .. import utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )
		
		with self.assertRaises( AssertionError ):
			disco1 = MockDiscovery.new_test( 'codecs' )
			disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = None )
		
		disco1 = MockDiscovery.new_test( 'codecs' )
		disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = 'codecs' )
		
		disco1 = MockDiscovery.new_test( 'codecs.utf8' )
		disco1.import_code( 'from . import utf8', Path( '__irrelevant__.py' ), scope = 'codecs.utf8' )

if __name__ == '__main__':
	logging.basicConfig( level = logging.DEBUG, force = True )
	unittest.main()
