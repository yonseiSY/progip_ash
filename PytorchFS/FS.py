from functools import reduce
import inspect
from typing import Union, Callable
import torch.nn as nn
import torch
import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class FS:
	def __init__(self) -> None:
		super().__init__()
		pass

	_log = []                   # For record information of all injection
	_recentPerturbation = None  # For Initialize accumalated faults when perform weight injection. targetLayer, singleDimensionalIdx, original value
	_neurons = []
	_NofNeurons = 0
	_layerInfo = None           # For optimize layer selection process. store kind of layer and it's indexes
	_isDone = False

	def getModuleByName(self, module, access_string):
		names = access_string.split(sep='.')
		return reduce(getattr, names, module)
	
	def getModuleNameList(self, module):
		moduleNames = []
		for name, l in module.named_modules():
			if not isinstance(l, nn.Sequential) and not isinstance(l, type(module)) and (name != ''):
          # print(name)
					moduleNames.append(name)
		return moduleNames
	
	def generateTargetIndexList(self, shape, n):
		result = []
		for i in range(n):
			tmp = []
			for i in shape:
				tmp.append(random.randint(0, i-1))
			result.append(tmp)
		return result
	
	def selectRandomTargetLayer(self, model, moduleNames, layerTypes=None):
		if(layerTypes == None):
			_targetIdx = random.randint(0, len(moduleNames)-1)
			_targetLayer = self.getModuleByName(model, moduleNames[_targetIdx])
			_targetLayerIdx = _targetIdx
		else:
			_targetLayerIdxCand = []  # Candidate of indexes, ex) conv2d = [0, 3, 6, 7] is appended.
			for x in layerTypes:
				if str(x) not in self._layerInfo:
					msg = f"This model has no attribute: {x}. You must set targetLayerTypes which belongs to {self._layerInfo.keys()}"
					raise KeyError(msg)
				else:
					_targetLayerIdxCand += self._layerInfo["{}".format(x)]

			# print(_targetLayerIdxCand)
			_targetIdxofCandList = random.randint(0, len(_targetLayerIdxCand)-1)
			# print(_targetLayerIdx, _targetLayerIdxCand[_targetLayerIdx])
			_targetLayer = self.getModuleByName(model, moduleNames[_targetLayerIdxCand[_targetIdxofCandList]])
			# print(_targetLayer.__class__)
			# print(_targetLayer, (type(_targetLayer) not in _layerFilter))
			_targetLayerIdx = _targetLayerIdxCand[_targetIdxofCandList]
		return _targetLayer, _targetLayerIdx
		

	def setLayerPerturbation(self, model: nn.Module):
		weights = model.features[0].weight.cpu().detach().numpy()
		weights.fill(0)
		model.features[0].weight = torch.nn.Parameter(torch.FloatTensor(weights).cuda())
	
	# def onlineNeuronInjection(model: nn.Module, targetLayer: str, NofTargetLayer: Union[list, int], targetNeuron: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
	# 	if(not((type(errorRate) == type(str)) ^ (type(NofError) == type(str)))):
	# 		raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
	# 	if(errorRate == "unset"):
	# 		_numError = NofError
	# 	if(NofError == "unset"):
	# 		_numError = 

	# 	if(targetLayer == "random"): # NofTargetLayer must be int
	# 		if(type(NofTargetLayer) != type(int)):
	# 			raise TypeError('Parameter "NofTargetLayer" must be int, when the value of parameter "targetLayer" is "random".')


	# 	return model

	def getLog(self):
		return self._log
	
	def getLastLog(self):
		return self._log[-1]
	
	def initLog(self):
		self._log = []


	def gatherAllNeuronValues(self, model: nn.Module, targetNeuron: str, targetLayer: str=None, targetLayerTypes: list=None):
		_moduleNames = self.getModuleNameList(model)
		_targetLayerIdxCand = []
		_hookHandlers = []

		def inputHook(module, input):
			# print(module)
			_neurons = input[0].cpu().detach().numpy()
			_singleDimensionalNeurons = _neurons.reshape(-1)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

		def outputHook(module, input, output):
			# print(module)
			_neurons = output.cpu().detach().numpy()
			_singleDimensionalNeurons = _neurons.reshape(-1)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

		if(targetLayer != None and targetLayerTypes == None):
			_targetLayerIdx = _moduleNames.index(targetLayer)
			_targetLayerIdxCand.append(_targetLayerIdx)

		if(targetLayer == None and targetLayerTypes != None):
			for x in targetLayerTypes:
				if str(x) not in self._layerInfo:
					msg = f"This model has no attribute: {x}. You must set targetLayerTypes which belongs to {self._layerInfo.keys()}"
					raise KeyError(msg)
				else:
					_targetLayerIdxCand += self._layerInfo["{}".format(x)]

		# print(_targetLayerIdxCand)

		if(targetNeuron=="output"):
			for idx in _targetLayerIdxCand:
				_hookHandlers.append(self.getModuleByName(model, _moduleNames[idx]).register_forward_hook(outputHook))
		elif(targetNeuron=="input"):
			for idx in _targetLayerIdxCand:
				_hookHandlers.append(self.getModuleByName(model, _moduleNames[idx]).register_forward_pre_hook(inputHook))
		else:
			raise ValueError("You must set 'targetNeuron' \"input\" or \"output\".")
		
		return _hookHandlers


	def setLayerInfo(self, model: nn.Module):
		self._layerInfo = dict()
		_moduleNames = self.getModuleNameList(model)
		for i in range(len(_moduleNames)):
			layer = self.getModuleByName(model, _moduleNames[i])
			if(str(type(layer)) not in self._layerInfo):
				self._layerInfo["{}".format((type(layer)))] = [i]
			else:
				self._layerInfo["{}".format((type(layer)))].append(i)
		
		# for i in self._layerInfo["{}".format((str(nn.modules.Conv2d)))]:
		# 	print(i)
		# 	print(self.getModuleByName(model, _moduleNames[i]))
	
	def onlineSingleLayerOutputInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", errorIdx: Union[list, str]="random", modifyValue: Union[Callable[[float], float], str]="None"):
		# print("Injection entered")
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input, output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			nonlocal _targetLayerIdx
			_neurons = output.cpu().detach().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)
			# plt.hist(_singleDimensionalNeurons, bins=100, range=[-2, 2])
			# plt.xlabel("Weight Value")
			# plt.ylabel("Count")
			# plt.show()


			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			# print(_neurons.shape)
			# print(_neurons.size)
			# print(_numError)

			if(errorIdx != "random"):
				_targetIndexes = errorIdx
			else:
				_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			# print(_targetIndexes)

			# print(targetBit)
			if(targetBit == "random"):
				_targetBitIdx = random.randint(0, 31)
			elif(type(targetBit) == int):
				_targetBitIdx = targetBit

			# print(_targetBitIdx)

			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				if(modifyValue == "None"):
					beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					beforeBinaryRep = binary(beforeDecRep)
					bits = list(beforeBinaryRep)
					bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
					afterBinaryRep = "".join(bits)
					_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)
					tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx[0], _targetBitIdx, beforeBinaryRep, beforeDecRep[0], afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx][0]))
				else:
					beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					beforeBinaryRep = binary(beforeDecRep)
					_singleDimensionalNeurons[_targetNeuronIdx] = modifyValue(_singleDimensionalNeurons[_targetNeuronIdx])
					afterDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					afterBinaryRep = binary(afterDecRep)
					tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx[0], "None", beforeBinaryRep, beforeDecRep[0], afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx][0]))



			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
		
			# self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			# self._NofNeurons += len(_singleDimensionalNeurons)

			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)
			
			output.data = torch.FloatTensor(_neurons).cuda()

			return output
		
		hookHandler = _targetLayer.register_forward_hook(hook)

		return hookHandler

	def onlineMultiBitLayerOutputInjection(self, model: nn.Module, targetLayer: str, bit_positions: list, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", errorIdx: Union[list, str]="random"):
		# print("Injection entered")
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')
  
		# 추가된 부분: 레이어의 총 뉴런 수 계산
		_targetLayer = self.getModuleByName(model, targetLayer)
		if hasattr(_targetLayer, 'weight'):
			total_neurons = _targetLayer.weight.numel()
		else:
			# 다른 방법으로 뉴런 수 계산
			# 예: forward hook 등을 사용
			total_neurons = 0  # 임시값  

		def hook(module, input, output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal _targetLayerIdx
			_neurons = output.cpu().detach().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)

			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			if(errorIdx != "random"):
				_targetIndexes = errorIdx
			else:
				_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			
			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
				beforeBinaryRep = binary(beforeDecRep)
				bits = list(beforeBinaryRep)
				
				for bit_pos in bit_positions:
					bits[bit_pos] = str(int(not bool(int(bits[bit_pos]))))
     
				afterBinaryRep = "".join(bits)
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)
    
				tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx[0], str(bit_positions), beforeBinaryRep, beforeDecRep[0], afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx][0]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)

			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)
    
			# 오류 주입 후 추가: 총 뉴런 수와 오류 주입된 뉴런 수를 저장
			self._current_injection_stats = {
				"injected_neurons": len(_targetIndexes),
				"total_neurons": _neurons.size,
				"ratio": len(_targetIndexes) / _neurons.size
			}      
			
			output.data = torch.FloatTensor(_neurons).cuda()

			return output
		
		hookHandler = _targetLayer.register_forward_hook(hook)

		return hookHandler
	
	def onlineSingleLayerInputInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", errorIdx: Union[list, str]="random", modifyValue: Union[Callable[[float], float], str]="None"):
		self._isDone = False
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			nonlocal _targetLayerIdx
			# print("Hook", self._isDone)
			# print(input)
			_neurons = input[0].cpu().detach().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)

			if(self._isDone):
				return torch.FloatTensor(_neurons).cuda()


			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			# print(_neurons.shape)
			# print(_neurons.size)
			# print(_numError)

			if(errorIdx != "random"):
				_targetIndexes = errorIdx
			else:
				_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			# print(_targetIndexes)

			# print(targetBit)
			if(targetBit == "random"):
				_targetBitIdx = random.randint(0, 31)
			elif(type(targetBit) == int):
				_targetBitIdx = targetBit

			# tmpLog = []
			# for _targetNeuronIdx in _targetIndexes:
			# 	beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
			# 	beforeBinaryRep = binary(beforeDecRep)
			# 	bits = list(beforeBinaryRep)
			# 	bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
			# 	afterBinaryRep = "".join(bits)
			# 	_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)

			# 	tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx]))
			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				if(modifyValue == "None"):
					beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					beforeBinaryRep = binary(beforeDecRep)
					bits = list(beforeBinaryRep)
					bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
					afterBinaryRep = "".join(bits)
					_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)
					tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx[0], _targetBitIdx, beforeBinaryRep, beforeDecRep[0], afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx][0]))
				else:
					beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					beforeBinaryRep = binary(beforeDecRep)
					_singleDimensionalNeurons[_targetNeuronIdx] = modifyValue(_singleDimensionalNeurons[_targetNeuronIdx])
					afterDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
					afterBinaryRep = binary(afterDecRep)
					tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx[0], "None", beforeBinaryRep, beforeDecRep[0], afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx][0]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
			
			# self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			# self._NofNeurons += len(_singleDimensionalNeurons)
			
			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)

			self._isDone = True

			return torch.FloatTensor(_neurons).cuda()
		
		hookHandler = _targetLayer.register_forward_pre_hook(hook)

		return hookHandler
	
	# def onlineMultiLayerOutputInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):


	def offlineSingleLayerWeightInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", accumulate: bool=True, errorIdx: Union[list, str]="random"):
		_moduleNames = self.getModuleNameList(model)
		# _moduleNames = [i for i in _moduleNames if "MaxPool2d" not in i or "ReLU" not in i]

		if(accumulate == False and self._recentPerturbation != None):  # Target of this method is SingleLayer, don't care of _recentPerturbation.targetLayerIdx = list
			# print("Recovery")
			# print(self._recentPerturbation)
			_recentTargetLayer = self.getModuleByName(model, _moduleNames[self._recentPerturbation["targetLayerIdx"]])
			_recentTargetWeights = _recentTargetLayer.weight.cpu().detach().numpy()
			_originalShape = _recentTargetWeights.shape
			_SDrecentTargetWeights = _recentTargetWeights.reshape(-1)
			for i in range(len(self._recentPerturbation["targetWeightIdxes"])):
				# print("("+str(i+1)+") " + str(_SDrecentTargetWeights[self._recentPerturbation["targetWeightIdxes"][i]]) + " -> " + str(self._recentPerturbation["originalValues"][i]))
				_SDrecentTargetWeights[self._recentPerturbation["targetWeightIdxes"][i]] = np.float64(self._recentPerturbation["originalValues"][i])
			
			_recentTargetLayer.weight = torch.nn.Parameter(torch.FloatTensor(_SDrecentTargetWeights.reshape(_originalShape)).cuda())
			
			self._recentPerturbation = None

		# _exceptLayers = [nn.modules.pooling, nn.modules.dropout, nn.modules.activation]
		# _layerFilter = tuple(x[1] for i in _exceptLayers for x in inspect.getmembers(i, inspect.isclass))
		# print(_layerFilter)


		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(type(_targetLayer))
		# print(_targetLayerIdx)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		_weights = _targetLayer.weight.cpu().detach().numpy()
		# print(_weights.shape)
		_originalWeightShape = _weights.shape
		_singleDimensionalWeights = _weights.reshape(-1)

		if(errorRate == "unset"):
				_numError = NofError
		if(NofError == "unset"):
			_numError = int(_weights.size * errorRate)

		if(errorIdx != "random"):
			_targetIndexes = errorIdx
		else:
			_targetIndexes = self.generateTargetIndexList(_singleDimensionalWeights.shape, _numError)
		
		if(targetBit == "random"):
			_targetBitIdx = random.randint(0, 31)
		elif(type(targetBit) == int):
			_targetBitIdx = targetBit

		_originalValues = []
		tmpLog = []
		for _targetWeightIdx in _targetIndexes:
			_originalValues.append(_singleDimensionalWeights[_targetWeightIdx])
			beforeDecRep = _singleDimensionalWeights[_targetWeightIdx]
			beforeBinaryRep = binary(beforeDecRep)
			bits = list(beforeBinaryRep)
			bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
			afterBinaryRep = "".join(bits)
			_singleDimensionalWeights[_targetWeightIdx] = np.float64(binToFloat(afterBinaryRep))
			tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetWeightIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalWeights[_targetWeightIdx]))
			
		self._recentPerturbation = {
				"targetLayerIdx": _targetLayerIdx,
				"targetWeightIdxes": _targetIndexes,
				"originalValues": _originalValues
			}
		if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
		else:
			self._log.append(tmpLog)

		_weights = _singleDimensionalWeights.reshape(_originalWeightShape)
		# print(_targetLayer.weight.cpu().detach().numpy() == _weights)
		_targetLayer.weight = torch.nn.Parameter(torch.FloatTensor(_weights).cuda())
		# torch.set_default_tensor_type(torch.cuda.DoubleTensor)
		# print(type(torch.cuda.DoubleTensor(_weights)))
		# _targetLayer.weight = torch.nn.Parameter(torch.DoubleTensor(_weights).cuda())

		# print(_singleDimensionalWeights)
		# print(len(_singleDimensionalWeights))
	
	def onlineSingleLayerBackwardInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", errorIdx: Union[list, str]="random"):
		self._isDone = False
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, grad_input, grad_output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			nonlocal _targetLayerIdx
			# print("Hook", self._isDone)
			# print(grad_input)
			_neurons = grad_input[0].cpu().detach().numpy()
			# print(np.asarray(grad_input[2].cpu()).shape)
			# print(grad_input[0])
			# print(type(torch.FloatTensor(grad_input[0].cpu().detach().numpy())))
			_originalNeuronShape = _neurons.shape
			# print(len(grad_input))
			# print(_originalNeuronShape)
			_singleDimensionalNeurons = _neurons.reshape(-1)


			if(self._isDone):
				return torch.FloatTensor([_neurons]).cuda()


			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			# print(_neurons.shape)
			# print(_neurons.size)
			# print(_numError)

			if(errorIdx != "random"):
				_targetIndexes = errorIdx
			else:
				_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			# print(_targetIndexes)

			# print(targetBit)
			if(targetBit == "random"):
				_targetBitIdx = random.randint(0, 31)
			elif(type(targetBit) == int):
				_targetBitIdx = targetBit

			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
				beforeBinaryRep = binary(beforeDecRep)
				bits = list(beforeBinaryRep)
				bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
				afterBinaryRep = "".join(bits)
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)

				tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
			
			# self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			# self._NofNeurons += len(_singleDimensionalNeurons)
			
			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)

			self._isDone = True

			# new_grad_input = np.asarray(grad_input.cpu().detach().numpy())

			# new_grad_input[0] = torch.FloatTensor(_neurons).cuda()
			return torch.FloatTensor([_neurons]).cuda()
		
		hookHandler = _targetLayer.register_full_backward_hook(hook)

		return hookHandler

	def onlineMultiBitLayerBackwardInjection(self, model: nn.Module, targetLayer: str, bit_positions: list, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", errorIdx: Union[list, str]="random"):
		self._isDone = False
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')
  
		# 추가된 부분: 레이어의 총 뉴런 수 계산
		_targetLayer = self.getModuleByName(model, targetLayer)
		if hasattr(_targetLayer, 'weight'):
			total_neurons = _targetLayer.weight.numel()
		else:
			# 다른 방법으로 뉴런 수 계산
			# 예: forward hook 등을 사용
			total_neurons = 0  # 임시값    

		def hook(module, grad_input, grad_output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal _targetLayerIdx
   
			_neurons = grad_input[0].cpu().detach().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)

			if(self._isDone):
				return torch.FloatTensor([_neurons]).cuda()

			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			if(errorIdx != "random"):
				_targetIndexes = errorIdx
			else:
				_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)

			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
				beforeBinaryRep = binary(beforeDecRep)
				bits = list(beforeBinaryRep)

				for bit_pos in bit_positions:
					bits[bit_pos] = str(int(not bool(int(bits[bit_pos]))))
     
				afterBinaryRep = "".join(bits)
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)

				tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx, str(bit_positions), beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
			
			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)
    
			# 오류 주입 후 추가: 총 뉴런 수와 오류 주입된 뉴런 수를 저장
			self._current_injection_stats = {
				"injected_neurons": len(_targetIndexes),
				"total_neurons": _neurons.size,
				"ratio": len(_targetIndexes) / _neurons.size
			}        

			self._isDone = True
   
			return torch.FloatTensor([_neurons]).cuda()
		
		hookHandler = _targetLayer.register_full_backward_hook(hook)

		return hookHandler