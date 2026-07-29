@cstruct
class RawEntry:
	hash: u64
	key_ptr: Ptr[None]
	value_ptr: Ptr[None]

@cstruct
class RawIndex:
	hash: u64
	entry_idx: usize

class RawDict:
	__entries: list[RawEntry]
	__indices: list[RawIndex]
	
	def __init__( self ) -> None:
		self.__entries = list[RawEntry]()
		self.__indices = list[RawIndex]()
	
	def _find_index_pos( self, hash: u64 ) -> usize:
		# Binary search via bisect for the entry inside sorted __indices
		return bisect.bisect_left( self.__indices, hash )
	
	def lookup( self, hash: u64, key_ptr: Ptr[None], key_eq_fn: ConstPtr[None] ) -> Result[Ptr[None], KeyError]:
		pos: usize = self._find_index_pos( hash )
		
		# Linear scan in case of hash collisions at position
		while pos < self.__indices.len():
			idx_node = self.__indices[pos]
			if idx_node.hash != hash:
				break
			
			entry = self.__entries[idx_node.entry_idx]
			# Call monomorphized equality function pointer
			if key_eq_fn( entry.key_ptr, key_ptr ):
				return Result.Ok( entry.value_ptr )
			
			pos += 1
			
		return Result.Err( KeyError )
	
	def insert( self, hash: u64, key_ptr: Ptr[None], value_ptr: Ptr[None], key_eq_fn: ConstPtr[None] ) -> None:
		pos: usize = self._find_index_pos( hash )
		
		# Check if key already exists (update path)
		scan_pos: usize = pos
		while scan_pos < self.__indices.len():
			idx_node = self.__indices[scan_pos]
			if idx_node.hash != hash:
				break
			entry = self.__entries[idx_node.entry_idx]
			if key_eq_fn( entry.key_ptr, key_ptr ):
				# Update existing value in place
				self.__entries[idx_node.entry_idx].value_ptr = value_ptr
				return
			scan_pos += 1
		
		# New key insertion path
		entry_idx: usize = self.__entries.len()
		self.__entries.append( RawEntry( hash = hash, key_ptr = key_ptr, value_ptr = value_ptr ) )
		
		# Maintain sorted order in __indices list
		self.__indices.insert( pos, RawIndex( hash = hash, entry_idx = entry_idx ) )

class dict[K, V]:
	__raw: RawDict
	
	def __init__( self ) -> None:
		self.__raw = RawDict()
	
	def __getitem__( self, key: K ) -> Result[V, KeyError]:
		h: u64 = hash( key )
		key_ptr: Ptr[None] = compiler.reinterpret_cast[Ptr[None]]( key )
		
		val_ptr = self.__raw.lookup( h, key_ptr, K.__eq_fn__ ).or_return()
		return Result.Ok( compiler.reinterpret_cast[V]( val_ptr ) )
	
	def __setitem__( self, key: K, value: V ) -> None:
		h: u64 = hash( key )
		key_ptr: Ptr[None] = compiler.reinterpret_cast[Ptr[None]]( key )
		val_ptr: Ptr[None] = compiler.reinterpret_cast[Ptr[None]]( value )
		
		self.__raw.insert( h, key_ptr, val_ptr, K.__eq_fn__ )
