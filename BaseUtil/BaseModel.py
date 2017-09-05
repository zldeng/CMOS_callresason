#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 14:26:45
'''
import sys
import os

import abc

class ModelBase(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self,config_file):
		self._config_file = config_file
	
	def _loadConfig(self,config_dict):
		return

	@abc.abstractmethod
	def loadModel(self):
		return

	@abc.abstractmethod
	def _saveModel(self):
		return 


	@abc.abstractmethod
	def trainLevel1TagModel(self):
		return
	
	@abc.abstractmethod
	def trainLevel2TagModel(self):
		return

	@abc.abstractmethod
	def predictLevel1TagForSamples(self,sample):
		return

	@abc.abstractmethod
	def predictLevel2TagForSamples(self,sampes):
		return

	@abc.abstractmethod
	def predictLevel1TagForSamplesWithProb(samples):
		return
	
	@abc.abstractmethod
	def predictLevel2TagForSamplesWithProb(samples):
		return

