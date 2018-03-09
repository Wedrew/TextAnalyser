import pyopencl as cl

## Find out about your computer's OpenCL situation and find device per platform
class getComputerInfo:
	def __init__(self):
		# Find out about your computer's OpenCL situation
		self.devices = []
		self.type = []
		print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
		# Print each platform on this computer
		for platform in cl.get_platforms():
			print('=' * 60)
			print('Platform - Name:  ' + platform.name)
			print('Platform - Vendor:  ' + platform.vendor)
			print('Platform - Version:  ' + platform.version)
			print('Platform - Profile:  ' + platform.profile)

		# Print each device per-platform
		for device in platform.get_devices():
			print('    ' + '-' * 56)
			print('    Device - Name:  ' + device.name)
			print('    Device - Type:  ' + cl.device_type.to_string(device.type))
			print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
			print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
			print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024))
			print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024))
			print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
			self.devices.append(device.name)
			self.type.append(cl.device_type.to_string(device.type))

	def getDevices(self):
			return self.devices

	def getTypes(self):
		return self.type

	def hasDiscreteCard(self):
		if self.type.count('GPU') >= 2:
			return True
		else:
			return False