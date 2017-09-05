#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 14:36:20
'''
import os
import sys
import ConfigParser

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

import pickle

reload(sys)
sys.setdefaultencoding('utf8')

from BaseModel import ModelBase



class LinearSvcClf(ModelBase):
	def __init__(self,config_dic):
		self._config_dic = config_dic

		#this is in para that need to be estimated before training a linearSvc model
		self._model_para_key = ['best__k','clf__C','clf__penalty','clf__tol','vec__max_df','vec__min_df','vec__ngram_range']

	def _loadConfig(self):
		try:
			self._model_para = {}
			for cand_key in self._model_para_key:
				self._model_para[cand_key] = self._config_dic['linearSvcModelPara'][cand_key.lower()]
				
			
			self._model_path = self._config_dic['modelPath']['path']
			
			self._level1_tag_model_name = os.path.join(self._model_path,self._config_dic['linearSvcModelPara']['level_1_tag_model_name'])
			self._level2_tag_model_name = os.path.join(self._model_path,self._config_dic['linearSvcModelPara']['level_2_tag_model_name'])

			return True
		except Exception,e:
			print 'Load linearSvc configure fail. err=' + str(e)
			sys.exit(1)
	

	def _saveModel(self,tag_level):
		try:
			if not os.path.exists(self._model_path):
				os.path.makedirs(self._model_path)
			
			if '1' == tag_level:
				pickle.dump(self._clf_linear_level1, file(self._level1_tag_model_name + '.model', 'wb'), True)
				pickle.dump(self._tf_vec_level1, file(self._level1_tag_model_name + '.vec', 'wb'), True)
				pickle.dump(self._best_level1, file(self._level1_tag_model_name + '.best', 'wb'), True)
			elif '2' == tag_level:
				pickle.dump(self._clf_linear_level2, file(self._level2_tag_model_name + '.model', 'wb'), True)
				pickle.dump(self._tf_vec_level2, file(self._level2_tag_model_name + '.vec', 'wb'), True)
				pickle.dump(self._best_level2, file(self._level2_tag_model_name + '.best', 'wb'), True)
				
			return True
		except Exception ,e:
			print 'save model fail. ' + str(e)
			return False



	def loadModel(self):
		
		if not self._loadConfig():
			print 'Load model  config file Fail'
			return False
		
		try:
			#load level1 model
			print 'begin load level1 model ...'
			self._tf_vec_level1 = pickle.load(file(self._level1_tag_model_name + '.vec','rb'))
			self._best_level1 = pickle.load(file(self._level1_tag_model_name + '.best','rb'))
			self._clf_linear_level1 = pickle.load(file(self._level1_tag_model_name + '.model','rb'))
		
			print 'Load level1 model done'
			

			#load level2 model
			self._tf_vec_level2 = pickle.load(file(self._level2_tag_model_name + '.vec','rb'))
			self._best_level2 = pickle.load(file(self._level2_tag_model_name + '.best','rb'))
			self._clf_linear_level2 = pickle.load(file(self._level2_tag_model_name + '.model','rb'))
		
			return True
		except Exception,e:
			print 'Load model fail. ' + str(e)
			return False

	def trainLevel1TagModel(self,level1_tag_train_x,level1_tag_train_y):
		'''
		train a linearSvc model
		train_x:[[x11,x12,x13,...],[x21,x22,x23...],...]
		train_y:[label_1,lable_2,...]
		'''
		if not self._loadConfig():
			print 'Load model  config file Fail'
			return False


		train_y_cnt = {}
		for cand_tag in level1_tag_train_y:
			train_y_cnt.setdefault(cand_tag,0)
			train_y_cnt[cand_tag] += 1
		
		deleted_tag = set()
		for cand_tag in train_y_cnt:
			if train_y_cnt[cand_tag] < 5:
				deleted_tag.add(cand_tag)
	
		new_train_x = []
		new_train_y = []
	
		for idx,cand_tag in enumerate(level1_tag_train_y):
			if cand_tag in deleted_tag:
				continue
	
			new_train_x.append(level1_tag_train_x[idx])
			new_train_y.append(level1_tag_train_y[idx])
		
	
		best_k = int(self._model_para['best__k'])
		clf_c = float(self._model_para['clf__C'])
		clf_penalty = self._model_para['clf__penalty']
		clf_tol = float(self._model_para['clf__tol'])
		vec_max_df = float(self._model_para['vec__max_df'])
		vec_min_df = int(self._model_para['vec__min_df'])
		vec__ngram_range_str = self._model_para['vec__ngram_range']

		try:
			vec__ngram_range_str = vec__ngram_range_str[1:-1]

			tmp_list = vec__ngram_range_str.strip().split(',')

			vec_ngram_range = (int(tmp_list[0]),int(tmp_list[1]))
		except Exception,e:
			print 'Load config fail.' + str(e)
			return False

		self._tf_vec_level1 = TfidfVectorizer(ngram_range=vec_ngram_range, min_df=vec_min_df, max_df=vec_max_df)
		self._best_level1 = SelectKBest(chi2, k=best_k)

		train_x_vec = self._tf_vec_level1.fit_transform(new_train_x)
		train_x_best = self._best_level1.fit_transform(train_x_vec,new_train_y)

		self._clf_linear_level1 = CalibratedClassifierCV(LinearSVC(C = clf_c,penalty = clf_penalty,tol = clf_tol))

		self._clf_linear_level1.fit(train_x_best,new_train_y)

		_tag_level = '1'
		if not self._saveModel(_tag_level):
			print 'Save level1 model Fail'
			return False

		return True
	

	def trainLevel2TagModel(self,level2_tag_train_x,level2_tag_train_y):
		'''
		train a linearSvc model
		train_x:[[x11,x12,x13,...],[x21,x22,x23...],...]
		train_y:[label_1,lable_2,...]
		'''
		if not self._loadConfig():
			print 'Load model  config file Fail'
			return False


		train_y_cnt = {}
		for cand_tag in level2_tag_train_y:
			train_y_cnt.setdefault(cand_tag,0)
			train_y_cnt[cand_tag] += 1
		
		deleted_tag = set()
		for cand_tag in train_y_cnt:
			if train_y_cnt[cand_tag] < 5:
				deleted_tag.add(cand_tag)
	
		new_train_x = []
		new_train_y = []
	
		for idx,cand_tag in enumerate(level2_tag_train_y):
			if cand_tag in deleted_tag:
				continue
	
			new_train_x.append(level2_tag_train_x[idx])
			new_train_y.append(level2_tag_train_y[idx])
		
	
		best_k = int(self._model_para['best__k'])
		clf_c = float(self._model_para['clf__C'])
		clf_penalty = self._model_para['clf__penalty']
		clf_tol = float(self._model_para['clf__tol'])
		vec_max_df = float(self._model_para['vec__max_df'])
		vec_min_df = int(self._model_para['vec__min_df'])
		vec__ngram_range_str = self._model_para['vec__ngram_range']

		try:
			vec__ngram_range_str = vec__ngram_range_str[1:-1]

			tmp_list = vec__ngram_range_str.strip().split(',')

			vec_ngram_range = (int(tmp_list[0]),int(tmp_list[1]))
		except Exception,e:
			print 'Load config fail.' + str(e)
			return False

		self._tf_vec_level2 = TfidfVectorizer(ngram_range=vec_ngram_range, min_df=vec_min_df, max_df=vec_max_df)
		self._best_level2 = SelectKBest(chi2, k=best_k)

		train_x_vec = self._tf_vec_level2.fit_transform(new_train_x)
		train_x_best = self._best_level2.fit_transform(train_x_vec,new_train_y)

		self._clf_linear_level2 = CalibratedClassifierCV(LinearSVC(C = clf_c,penalty = clf_penalty,tol = clf_tol))

		self._clf_linear_level2.fit(train_x_best,new_train_y)

		_tag_level = '2'
		if not self._saveModel(_tag_level):
			print 'Save level2 model Fail'
			return False

		return True

	def predictLevel1TagForSamples(self,samples):
		'''
		input:
			samples: [[x11,x12,x13,...],[x21,x22,x23,...],...]
		
		output:
			test_y: [y1,y1,...]

		'''
		self._labels_list = list(self._clf_linear_level1.classes_)

		vec_test = self._tf_vec_level1.transform(samples)
		best_test = self._best_level1.transform(vec_test)

		predict_result = self._clf_linear_level1.predict(best_test)
	
		return predict_result

	def predictLevel2TagForSamples(self,samples):
		self._labels_list = list(self._clf_linear_level2.classes_)

		vec_test = self._tf_vec_level2.transform(samples)
		best_test = self._best_level2.transform(vec_test)

		predict_result = self._clf_linear_level2.predict(best_test)
	
		return predict_result

	def predictLevel1TagForSamplesWithProb(self,samples):
		'''
		input:
			samples: [[x11,x12,x13,...],[x21,x22,x23,...],...]
		
		output: return each label and it's prob
			test_y: [
						[(y11,p11),(y12,p12),..],
						[(y21,p21),(y22,p22),..],
						...]
		'''
		self._labels_list = list(self._clf_linear_level1.classes_)

		vec_test = self._tf_vec_level1.transform(samples)
		best_test = self._best_level1.transform(vec_test)

		predict_prob = self._clf_linear_level1.predict_proba(best_test)
		
		predict_prob_result = []

		for idx in range(len(samples)):
			predict_prob_list = predict_prob[idx]
			
			label2prob = {}
			for prob_idx in range(len(predict_prob_list)):
				label2prob[self._labels_list[prob_idx]] = predict_prob_list[prob_idx]

			sorted_list = sorted(label2prob.items(),key=lambda d: d[1],reverse=True)

			predict_prob_result.append(sorted_list)

		return predict_prob_result
				
	def predictLevel2TagForSamplesWithProb(self,samples):
		'''
		input:
			samples: [[x11,x12,x13,...],[x21,x22,x23,...],...]
		
		output: return each label and it's prob
			test_y: [
						[(y11,p11),(y12,p12),..],
						[(y21,p21),(y22,p22),..],
						...]
		'''
		self._labels_list = list(self._clf_linear_level2.classes_)

		vec_test = self._tf_vec_level2.transform(samples)
		best_test = self._best_level2.transform(vec_test)

		predict_prob = self._clf_linear_level2.predict_proba(best_test)
		
		predict_prob_result = []

		for idx in range(len(samples)):
			predict_prob_list = predict_prob[idx]
			
			label2prob = {}
			for prob_idx in range(len(predict_prob_list)):
				label2prob[self._labels_list[prob_idx]] = predict_prob_list[prob_idx]

			sorted_list = sorted(label2prob.items(),key=lambda d: d[1],reverse=True)

			predict_prob_result.append(sorted_list)

		return predict_prob_result
