#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-24 15:17:27
'''
 
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('./LinearSvc')
sys.path.append('./Fasttext')
sys.path.append('./BaseUtil')


from DataUtil import loadConfig
from DataUtil import convertXmlDataToSeggedData

from LinearSvcClf import LinearSvcClf
from FastTextClf import FastTextClf

from optparse import OptionParser

def classifyXmlDataFile(config_file,xml_data_file,out_result_file):
	'''
	select fasttext as classify tool.
	No merge
	'''
	config_dic = loadConfig(configure_file)
	
	fasttext_clf = FastTextClf(config_dic)

	if not fasttext_clf.loadModel():
		print 'Load fasttext model fail'
		sys.exit(1)

	linear_svc_clf = LinearSvcClf(config_dic)
	if not linear_svc_clf.loadModel():
		print 'Load svc model fail'
		sys.exit(1)

	case_id2data,case_id2seg_data = convertXmlDataToSeggedData(xml_data_file)

	case_id_list = []
	data_list = []

	for cand_id in case_id2seg_data:
		case_id_list.append(cand_id)
		data_list.append(case_id2seg_data[cand_id])

	level1_predict = fasttext_clf.predictLevel1TagForSamplesWithProb(data_list)
	level2_predict = linear_svc_clf.predictLevel2TagForSamplesWithProb(data_list)

	if len(case_id_list) != len(level1_predict) \
		or len(level1_predict) != len(level2_predict):
		print 'Predict data fail. result length error:' + iflytek_in_result_file
		sys.exit(1)
	
	out_file = file(out_result_file,'w')
	

	for idx in range(len(case_id_list)):
		level_1_tag_pred_list = level1_predict[idx]
		level_2_tag_pred_list = level2_predict[idx]

		pred_level_1_tag = level_1_tag_pred_list[0][0]
		pred_level_2_tag = level_2_tag_pred_list[0][0]

		#根据一级分类标签对二级分类标签进行修正
		#按照二级分类排序，选择第一个对应的一级分类标签和预测的一级分类标签一致的二级分类标签作为最终二级标签
		changed_level_2_tag = pred_level_2_tag
		for cand_tag_prob_pair in level_2_tag_pred_list:
			cand_pred_level_2_tag = cand_tag_prob_pair[0]

			if cand_pred_level_2_tag in level2_tag2level1_tag \
				and level2_tag2level1_tag[cand_pred_level_2_tag] == pred_level_1_tag:
					changed_level_2_tag = cand_pred_level_2_tag
					break
		
		level_1_tag = pred_level_1_tag
		level_2_tag = changed_level_2_tag

		case_id = case_id_list[idx]
		
		case_data = case_id2data.get(case_id,None)
		if case_data is None:
			print 'No data for : ' + case_id
			continue

		
		result = case_id + '\t' + level_1_tag + '\t' + level_2_tag + '\t' + case_data
		out_file.write(result + '\n')

	out_file.close()


def usage():
	print 'prama_config.py usage:'
	print '-c, --config: config_file.'
	print '-i, input: in_result_file[abs_path]'
	print '-o, --output: out_file[abs_path]'

if __name__ == '__main__':
	'''
	configure_file = '/home/dengzhilong/work/call_reason/sklearn_code/Model/Code/henan_callreason.cfg'
	result_in_file = '/home/dengzhilong/work/call_reason/sklearn_code/Model/Code/data/tmp.txt'
	out_file = result_in_file + '.out'
	'''
	
	parser = OptionParser()
	parser.add_option('-c','--config',dest='config_file',action='store',type='string',help='model configure file')
	parser.add_option('-i','--input',dest='in_file',action='store',type='string',help='input xml_data file')
	parser.add_option('-o','--output',dest='out_file',action='store',type='string',help='classify result file')

	options,args = parser.parse_args()
	
	configure_file = options.config_file
	xml_in_file = options.in_file
	out_file = options.out_file

	if configure_file is None or xml_in_file is None or out_file is None:
		parser.print_help()
		sys.exit(1)

	classifyXmlDataFile(configure_file,xml_in_file,out_file)
