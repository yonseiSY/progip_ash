import struct

def binary(num):
		return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def binToFloat(num):
 f = int(num, 2)
 return struct.unpack('f', struct.pack('I', f))[0]

def prod(val):
 res = 1
 for ele in val:
  res *= ele
 return res