#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 16:07:59
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('./LinearSvc')
sys.path.append('./Fasttext')
sys.path.append('./BaseUtil')



from DataUtil import loadTrainTestSkData
from DataUtil import CmpResult
from DataUtil import CmpListWithProb
from DataUtil import classifyUseTwoModelsAndMerge

from DataUtil import loadConfig
from DataUtil import loadSkData

from LinearSvcClf import LinearSvcClf
from FastTextClf import FastTextClf



def trainFastTextModel(train_sk_file,config_dic):
	level1_train_x,level1_train_y,level1_other_info_list = loadSkData(train_sk_file,'1')
	level2_train_x,level2_train_y,level2_other_info_list = loadSkData(train_sk_file,'2')
	
	fasttext_clf = FastTextClf(config_dic)

	train_level1 = fasttext_clf.trainLevel1TagModel(level1_train_x,level1_train_y)
	train_level2 = fasttext_clf.trainLevel2TagModel(level2_train_x,level2_train_y)

	if not train_level1 or not train_level2:
		print 'Train FastText model fail'
	else:
		print 'Train FastText model done!'


def testFastText(config_dic,test_sk_file):
	fasttext_clf = FastTextClf(config_dic)

	level1_test_x,level1_test_y,level1_other_info_list = loadSkData(test_sk_file,'1')
	level2_test_x,level2_test_y,level2_other_info_list = loadSkData(test_sk_file,'2')

	if not fasttext_clf.loadModel():
		print 'FastText load Model Fail'
		sys.exit(1)

	level1_predict_prob_result = fasttext_clf.predictLevel1TagForSamplesWithProb(level1_test_x)

	fasttext_level1_result = 'henan_level1_fasttext.result'
	if CmpListWithProb(level1_other_info_list,level1_test_y,
		level1_predict_prob_result,fasttext_level1_result) is None:
		print  'CmpListWithProb fasttext level1 fail'
		sys.exit(1)

	level2_predict_prob_result = fasttext_clf.predictLevel2TagForSamplesWithProb(level2_test_x)

	fasttext_level2_result = 'henan_level2_fasttext.result'
	if CmpListWithProb(level2_other_info_list,level2_test_y,
		level2_predict_prob_result,fasttext_level2_result) is None:
		print 'CmpListWithProb fasttext level2 fail'

		sys.exit(1)

	print 'test Fasttext done!'

	

def trainLinearSvcModel(train_sk_file,config_dic):
	level1_train_x,level1_train_y,level1_other_info_list = loadSkData(train_sk_file,'1')
	level2_train_x,level2_train_y,level2_other_info_list = loadSkData(train_sk_file,'2')

	linear_svc_clf = LinearSvcClf(config_dic)

	train_level1 = linear_svc_clf.trainLevel1TagModel(level1_train_x,level1_train_y)
	train_level2 = linear_svc_clf.trainLevel2TagModel(level2_train_x,level2_train_y)

	if not train_level1 or not train_level2:
		print 'Train LinearSvc model fail'
		sys.exit(1)
	else:
		print 'Train LinearSvc model done'

def testLinearSvcModel(test_sk_file,config_dic):
	linear_svc_clf = LinearSvcClf(config_dic)

	level1_test_x,level1_test_y,level1_other_info_list = loadSkData(test_sk_file,'1')
	level2_test_x,level2_test_y,level2_other_info_list = loadSkData(test_sk_file,'2')

	
	if not linear_svc_clf.loadModel():
		print 'Load LinearSvc model fail'
		sys.exit(1)

	level1_predict_prob_result = linear_svc_clf.predictLevel1TagForSamplesWithProb(level1_test_x)

	linear_svc_level1_result = 'henan_level1_svc.result'

	if CmpListWithProb(level1_other_info_list,level1_test_y,
		level1_predict_prob_result,linear_svc_level1_result) is None:
			print 'CmpListWithProb linearSvc level1 fail'
			sys.exit(1)
	
	level2_predict_prob_result = linear_svc_clf.predictLevel2TagForSamplesWithProb(level2_test_x)
	linear_svc_level2_result = 'henan_level2_svc.result'
	
	if CmpListWithProb(level2_other_info_list,level2_test_y,
		level2_predict_prob_result,linear_svc_level2_result) is None:
			print 'CmpListWithProb linearSvc level2 fail'
			sys.exit(1)


	print 'Test linearSvc done!'

'''
#todos
def testMerge():
	fasttext_config_file = '/home/dengzhilong/work/call_reason/sklearn_code/Model/Fasttext/fasttext.cfg'
	svc_config_file = '/home/dengzhilong/work/call_reason/sklearn_code/Model/LinearSvc/linear_svc.cfg'
	train_sk_file = data_dir + 'all_train.sk'
	test_sk_file = data_dir + 'henan_1th_all_labeled_data_from_excel.available.test.sklearn'
	tag_level = '2'
	result_file = 'all_svc_fasttext_merge_' + tag_level + '.result'
	
	is_training = True
	
	linear_svc_cls = LinearSvcClf(svc_config_file)
	fasttext_cls = FastTextClf(fasttext_config_file)

	if is_training:
		train_x,train_y,other_info_list = loadSkData(train_sk_file,tag_level)
		
		if not linear_svc_cls.trainModel(train_x,train_y):
			print 'Train Svc model fail'
			sys.exit(1)

		if not fasttext_cls.trainModel(train_x,train_y):
			print 'Train fasttext model fail'
			sys.exit(1)

	if not linear_svc_cls.init_flag:
		if not linear_svc_cls.loadModel():
			print 'Load linear model fail'
			sys.exit(1)


	if not fasttext_cls.init_flag:
		if not fasttext_cls.loadModel():
			print 'Load fasttext model fail'
			sys.exit(1)

	test_x,test_y,other_info_list = loadSkData(test_sk_file,tag_level)
	svc_prob_result = linear_svc_cls.predictSamplesWithProb(test_x)
	fasttext_prob_result = fasttext_cls.predictSamplesWithProb(test_x)

	
	if not classifyUseTwoModelsAndMerge(test_y,other_info_list,result_file,
		svc_prob_result,fasttext_prob_result):
			print 'Merge model fail'
			sys.exit(1)
'''	




if __name__ == '__main__':
	configure_file = '/home/dengzhilong/work/call_reason/sklearn_code/Model/Code/henan_callreason.cfg'
	data_dir = '/home/dengzhilong/work/call_reason/data/all_data/'
	train_sk_file = data_dir + 'all_hn_train_0817.sk'
	test_sk_file = data_dir + 'henan_1th_all_labeled_data_from_excel.available.test.sklearn'

	conf_dic = loadConfig(configure_file)
	print conf_dic
	
	#trainFastTextModel(train_sk_file,conf_dic)
	#testFastText(conf_dic,test_sk_file)
	

	trainLinearSvcModel(train_sk_file,conf_dic)
	testLinearSvcModel(test_sk_file,conf_dic)
	
