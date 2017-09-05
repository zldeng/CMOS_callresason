#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 18:59:13
'''
import abc
import ConfigParser
import fasttext
import sys
import os

from BaseModel import ModelBase
from DataUtil import loadTrainTestSkData
from DataUtil import convertSkTrainFileToFastTextFile

class FastTextClf(ModelBase):
	def __init__(self,config_dic):
		self._config_dic = config_dic

	def _loadConfig(self):
		try:
			self._model_path = self._config_dic['modelPath']['path']
			
			self._level1_tag_model_name = os.path.join(self._model_path,	
				self._config_dic['fasttextModelPara']['level_1_tag_model_name'])
			
			self._level2_tag_model_name = os.path.join(self._model_path,
				self._config_dic['fasttextModelPara']['level_2_tag_model_name'])

			self._label_prefix = self._config_dic['fasttextModelPara']['label_prefix']
		except Exception,e:
			print 'Load fasttext configure fail. err = ' + str(e)
			sys.exit(1)
			#return False

		return True
	
		
	def loadModel(self):
		try:
			if not self._loadConfig():
				print 'Load model config fail'
				return False

			self._fasttext_level1_model = fasttext.load_model(self._level1_tag_model_name + '.bin',encoding='utf8')
			self._fasttext_level2_model = fasttext.load_model(self._level2_tag_model_name + '.bin',encoding='utf8')
			
			return True
		except Exception,e:
			print 'Load model fail. err=' + str(e)
			sys.exit(1)
			#return False

	def _saveModel(self):
		pass
	

	def trainLevel1TagModel(self,level1_train_x,level1_train_y):
		return self._trainModel(level1_train_x,level1_train_y,'1')
	
	def trainLevel2TagModel(self,level2_train_x,level2_train_y):
		return self._trainModel(level2_train_x,level2_train_y,'2')


	def _trainModel(self,train_x,train_y,tag_level):
		if not self._loadConfig():
			sys.exit(1)

		#create tmp fasttext train file from sklearn train file
		train_file_name = convertSkTrainFileToFastTextFile(train_x,train_y,self._model_path,self._label_prefix)
		if train_file_name is None:
			print 'convert train file fail'
			sys.exit(1)
			#return False
		
		try:
			if '1' == tag_level:
				model_name = self._level1_tag_model_name
			elif '2' == tag_level:
				model_name = self._level2_tag_model_name
			else:
				print 'Error tag_level: ' + tag_level
				sys.exit(1)

			self._fasttext = fasttext.supervised(train_file_name,model_name,
				label_prefix = self._label_prefix)
			
		except Exception,e:
			print 'train fasttext model fail. ' + str(e)
			sys.exit(1)
		
		#delete tmp train file
		os.remove(train_file_name)
		
		return True
	

	def predictLevel1TagForSamples(self,samples):
		return self._predictSamples(samples,'1')
	
	def predictLevel2TagForSamples(self,samples):
		return self._predictSamples(samples,'2')

	def _predictSamples(self,samples,tag_level):
		predict_data = []
		
		if '1' == tag_level:
			tmp_predict_result = self._fasttext_level1_model.predict(samples)
		elif '2' == tag_level:
			tmp_predict_result = self._fasttext_level2_model.predict(samples)
		else:
			print 'Error tag_level: ' + tag_level
			sys.exit(1)

		predict_result = []

		for cand_label in tmp_predict_result:
			cand_pred_tag = cand_label[0].replace(self._label_prefix,'').encode('utf8')
			predict_result.append(cand_pred_tag)

		return predict_result

	def predictLevel1TagForSamplesWithProb(self,samples):
		return self._predictSamplesWithProb(samples,'1')
	
	def predictLevel2TagForSamplesWithProb(self,samples):
		return self._predictSamplesWithProb(samples,'2')
	
	def _predictSamplesWithProb(self,samples,tag_level):
		if '1' == tag_level:
			_fasttext = self._fasttext_level1_model
		elif '2' == tag_level:
			_fasttext = self._fasttext_level2_model

		#get topK labels prob
		top_labels_cnt = 10

		predict_prob_result = _fasttext.predict_proba(samples,top_labels_cnt)
		
		prob_result = []
		for idx in range(len(predict_prob_result)):
			tmp_list = []
			for idx_2 in range(len(predict_prob_result[idx])):
				tmp_list.append((predict_prob_result[idx][idx_2][0].replace(self._label_prefix,'').encode('utf8'),predict_prob_result[idx][idx_2][1]))
			
			prob_result.append(tmp_list)

		return prob_result


