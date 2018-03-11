import pyopencl as cl

class GetComputerInfo:
	def __init__(self):
		# Find out about your computer's OpenCL situatio
		self.devicesList = []
		self.typeList = []
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
			self.devicesList.append(device.name)
			self.typeList.append(cl.device_type.to_string(device.type))

		print('=' * 60)
		self.platform = cl.get_platforms()[0]

		if self.typeList.count('GPU') == 2:
			self.device = platform.get_devices()[2]
			self.context = cl.Context([self.device])
			self.queue = cl.CommandQueue(self.context)
			print("Discrete graphics card detected!")
			print("Setting {} as primary device\n".format(self.device.name))

		elif self.typeList.count('GPU') == 1:
			self.device = platform.get_devices()[1]
			self.context = cl.Context([self.device])
			self.queue = cl.CommandQueue(self.context)
			print("No discrete graphics card detected!")
			print("Setting {} as primary device\n".format(self.device.name))

		else:
			self.device = platform.get_devices()[0]
			self.context = cl.Context([self.device])
			self.queue = cl.CommandQueue(self.context)
			print("Only cpu detected!")
			print("Setting {} as primary device\n".format(self.device.name))

	def getPlatform(self):
		return self.platform

	def getDevices(self):
		return self.device

	def getContext(self):
		return self.context

	def getQueue(self):
		return self.queue