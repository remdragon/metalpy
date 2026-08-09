def foo( hello: str, world: str ):
	hello_world = hello + ' ' + world
	print( hello_world.lower() )

def main():
	foo( 'Hello', 'MetalPy' )
