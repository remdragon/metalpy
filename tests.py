# stdlib imports:
import sys
import unittest

if __name__ == '__main__':
	loader = unittest.TestLoader()
	suite = loader.discover( start_dir = '.', pattern = '*_test.py' )
	result = unittest.TextTestRunner( verbosity = 1 ).run( suite )
	sys.exit( 0 if result.wasSuccessful() else 1 )
