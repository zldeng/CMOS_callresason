#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 16:14:39
'''
import os
import sys
import json
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('/home/dengzhilong/work/call_reason/sklearn_code/Model/Code/LtpTool/')

import ConfigParser

from LtpSegPosTool import ltpSegPos

def loadConfig(config_file):
	try:
		conf = ConfigParser.ConfigParser()
		conf.read(config_file)

		section_list = conf.sections()

		conf_dic = {}

		for cand_section in section_list:
			conf_dic[cand_section] = {}
			options_list = conf.options(cand_section)
			
			for cand_opt in options_list:
				conf_dic[cand_section][cand_opt] = conf.get(cand_section,cand_opt)
		
		model_dir = conf_dic['modelPath']['path']

		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		
		return conf_dic
	except Exception,e:
		print 'Load configure file error. err = ' + str(e)
		sys.exit(1)

def getTagPairInfo(conf_dic):
	level2_tag2level1_tag = {}
	level1_tag2level2_tag_set = {}
	
	id2level1_tag = {}
	id2level2_tag = {}
	for cand_tag in conf_dic['classifyTag']:
		cand_id = conf_dic['classifyTag'][cand_tag]

		if len(cand_id) < 2:
			id2level1_tag[cand_id] = cand_tag
		else:
			id2level2_tag[cand_id] = cand_tag
	
	tag_pair_list = []
	for cand_id in id2level2_tag:
		cand_level2_tag = id2level2_tag[cand_id]

		level1_tag_id = cand_id[0]
		if level1_tag_id not in id2level1_tag:
			print 'Err_tag: ' + level1_tag_id + '\t' + cand_id
			sys.exit(1)

		cand_level1_tag = id2level1_tag[level1_tag_id]

		tag_pair_list.append((cand_level1_tag,cand_level2_tag))


	for tag_pair in tag_pair_list:
		level1_tag = tag_pair[0]
		level2_tag = tag_pair[1]

		level2_tag2level1_tag[level2_tag] = level1_tag

		level1_tag2level2_tag_set.setdefault(level1_tag,set())

		level1_tag2level2_tag_set[level1_tag].add(level2_tag)
	
	return level2_tag2level1_tag,level1_tag2level2_tag_set
	


def loadSkData(sk_file_name,tag_level):
	info_data_list,level1_tag_list,level2_tag_list,other_info_list = loadTrainTestSkData(sk_file_name)

	data_x = info_data_list
	
	if '1' == tag_level:
		data_y = level1_tag_list
	elif '2' == tag_level:
		data_y = level2_tag_list
	else:
		print 'Error tag_level: ' + tag_level
		sys.exit(1)

	return data_x,data_y,other_info_list
	

def loadTrainTestSkData(sklearn_data):
	'''
	将分好词的标注问题转换成sklearn可直接使用的语料
	input:excel_name section_name tag1 tag2 word1 word2....
	train/test:
	'''

	info_data_list = []
	level1_tag_list = []
	level2_tag_list = []
	
	other_info_list = []
	for line in file(sklearn_data):
		line_list = line.strip().split(' ')

		info_data = ' '.join(line_list[4:])
		level_1_tag = line_list[2].strip()
		level_2_tag = line_list[3].strip()

		if level_1_tag == '' or level_2_tag == '':
			continue

		info_data_list.append(info_data)
		level1_tag_list.append(level_1_tag)
		level2_tag_list.append(level_2_tag)

		other_info_list.append(line_list[0] + ' ' + line_list[1])

	return info_data_list,level1_tag_list,level2_tag_list,other_info_list
	
def convertSkTrainFileToFastTextFile(train_x,train_y,model_path,label_prefix):
	try:
		if not os.path.exists(model_path):
			os.path.makedirs(model_path)

		import time
		tmp_train_file = str(time.time()) + '.fasttext_train'
		train_file_name = os.path.join(model_path,tmp_train_file)

		fasttext_train_file = file(train_file_name,'w')
		for cand_idx in range(len(train_x)):
			cand_data = train_x[cand_idx] + '\t' + label_prefix + train_y[cand_idx]

			fasttext_train_file.write(cand_data + '\n')

		fasttext_train_file.close()

		return train_file_name
	except Exception,e:
		print 'Convert train file fail. ' + str(e)
		return None
	
	
def CmpResult(other_info_list,gold_label_list,predict_label_list,result_file):
	if len(gold_label_list) != len(predict_label_list):
		print 'List not same length'
		return None
	
	out_file = file(result_file,'w')
	

	total = len(gold_label_list)
	
	correct = 0
	for idx in range(total):
		if gold_label_list[idx] == predict_label_list[idx]:
			result = 'good_case:'
			correct += 1
		else:
			result = 'bad_case:'

		result += '\t' + other_info_list[idx] + '\t' + gold_label_list[idx] + '\t' + predict_label_list[idx]

		out_file.write(result + '\n')
	

	out_file.write('\n\n' + str(total) + '\t' + str(correct) + '\t' + str(correct * 1.0 / total) + '\n')
	
	out_file.close()
	
	return True


def CmpListWithProb(other_info_list,gold_label_list,predict_label_with_prob,result_file):
	if len(gold_label_list) != len(predict_label_with_prob):
		print 'list not same: ' + str(len(gold_label_list)) + '\t' + str(len(predict_label_with_prob))
		return None

	out_file = file(result_file,'w')

	total = len(gold_label_list)

	correct = 0

	import json
	for idx in range(total):
		if gold_label_list[idx] == predict_label_with_prob[idx][0][0]:
			result = 'good_case:'
			correct += 1
		else:
			result = 'bad_case:'

		result += '\t' + other_info_list[idx] + '\t' + gold_label_list[idx] + '\t' + json.dumps(predict_label_with_prob[idx],ensure_ascii = False)

		out_file.write(result + '\n')
	
	out_file.write('\n\n' + str(total) + '\t' + str(correct) + '\t' + str(correct * 1.0 / total) + '\n')
	
	out_file.close()
	
	
	return True




def classifyUseTwoModelsAndMerge(test_y,other_info_list,result_file,
	svc_prob_result,fasttext_prob_result):
	out_file = file(result_file,'w')

	if len(svc_prob_result) != len(fasttext_prob_result):
		print 'result len not same: ' + str(le(svc_prob_result)) + ' : ' + str(len(fasttext_prob_result))
		return False
	
	total_samples = len(test_y)
	svc_correct_samples = 0
	fasttext_correct_samples = 0
	merge_correct_samples = 0

	for idx in range(total_samples):
		gold_label = test_y[idx]

		svc_predict = svc_prob_result[idx]
		fasttext_predict = fasttext_prob_result[idx]

		merge_predict_label = svc_predict[0][0]
		
		svc_prob_diff = svc_predict[0][1] - svc_predict[1][1]

		fasttext_prob_diff = fasttext_predict[0][1] - fasttext_predict[1][1]

		if fasttext_prob_diff > svc_prob_diff:
			merge_predict_label = fasttext_predict[0][0]
		
		if gold_label == svc_predict[0][0]:
			svc_correct_samples += 1

		if fasttext_predict[0][0] == gold_label:
			fasttext_correct_samples += 1

		
		if gold_label == merge_predict_label:
			merge_correct_samples += 1
			merge_result = 'good_case:'
		else:
			merge_result = 'bad_case:'

		merge_result += '\t' + other_info_list[idx] + '\t' + gold_label \
			+ '\t' + merge_predict_label \
			+ '\t' + json.dumps(svc_prob_result[idx],ensure_ascii = False) \
			+ '\t' + json.dumps(fasttext_prob_result[idx],ensure_ascii = False)

		out_file.write(merge_result + '\n')

	out_file.write('total: ' + str(total_samples) + '\n')
	out_file.write('svc: ' + str(svc_correct_samples) + '\t' + str(svc_correct_samples * 1.0 / total_samples) + '\n')
	out_file.write('fasttext: ' + str(fasttext_correct_samples) + '\t' + str(fasttext_correct_samples * 1.0 / total_samples) + '\n')
	out_file.write('megre: ' + str(merge_correct_samples) + '\t' + str(merge_correct_samples * 1.0 /total_samples) + '\n')

	out_file.close()
	return True




def loadDataFromIflytekResultFile(iflytek_result_file):
	'''
	load data from iflytek_result file and return data transformed data
	return: {id:[{'type':'c','data':'xxxxx'}]}
	'''
	
	iflytek_result = {}

	line_num = 0
	for line in file(iflytek_result_file):
		line_num += 1
		line = line.strip()
		if line == '流水号$路径$时长$格式$文本$时间$声道$文本长度$静音$有效时长':
			continue

		line_list = line.split('$')

		if len(line_list) != 10:
			print 'Error line ' + iflytek_result_file + ' ' + str(line_num)
			continue

		case_id = line_list[0]

		word_list = line_list[4].strip().split(';')

		voice_id_list = line_list[6].strip().split(';')

		if len(word_list) != len(voice_id_list):
			print 'data_cnt_error:' +  + iflytek_result_file + ' ' + str(line_num)
			continue

		data_list = []
		
		data_str = ''
		voice_id_str = ''
		for idx in range(len(voice_id_list)):
			cand_id = voice_id_list[idx]
			cand_word = word_list[idx]

			if data_str == '':
				data_str = cand_word
				voice_id_str = cand_id
			else:
				#start a new sentcence
				if cand_id != voice_id_str:
					data_dic = {'data':data_str}
					data_dic['type'] = 'w' if voice_id_str == '0' else 'c'

					data_list.append(data_dic)
					
					data_str = ''
					voice_id_str = ''

				data_str += cand_word
				voice_id_str = cand_id

		if '' != data_str:
			data_dic = {'data':data_str}
			data_dic['type'] = 'w' if voice_id_str == '0' else 'c'

			data_list.append(data_dic)

		
		if len(data_list) > 0:
			#result_data = dir_name + '\t' + case_id + '\t' + 'tag_1\ttag_2' + '\t' + json.dumps(data_list,ensure_ascii = False)
			#out_file.write(result_data + '\n')

			iflytek_result[case_id] = data_list
		
	print 'load result data from ' + iflytek_result_file + ' get cases: ' + str(len(iflytek_result))
	return iflytek_result



def segData(iflytek_result_data):
	'''
	seg data from iflytek_result_data
	input: {id:[{'type':'c','data':'xxxxx'}]}
	output:{id:'word1 word2 word3...',id: 'word1 word2 ...'} 
	'''

	seg_result = {}

	pos_black_set = ['wp'] #punctuation is dropped
	for cand_id in iflytek_result_data:
		cand_data_list = iflytek_result_data[cand_id]
		
		seg_list = []

		for tmp_dic in cand_data_list:
			cand_type = tmp_dic['type']
			cand_data = tmp_dic['data'].strip()

			if cand_data == '':
				continue
			
			if isinstance(cand_data,unicode):
				cand_data = cand_data.encode('utf8')

			#print 'data: ' + cand_data
			#print type(cand_data)
			try:
				tmp_seg,tmp_pos = ltpSegPos(cand_data)
			except Exception,e:
				print 'ltp_error: ' + str(e)
				continue
			
			if None == tmp_seg or tmp_pos == None:
				continue
			
			for cand_idx in range(len(tmp_seg)):
				cand_word = tmp_seg[cand_idx]
				cand_pos = tmp_seg[cand_idx]

				if cand_pos in pos_black_set:
					continue
				
				seg_list.append(cand_word)
		
		if len(seg_list) > 0:
			seg_result[cand_id] = ' '.join(seg_list)
	
	return seg_result


def convertDataFromIflytekResultFileToSeggedData(iflytek_result_file_name):
	iflytek_transfor_result = loadDataFromIflytekResultFile(iflytek_result_file_name)

	seg_result = segData(iflytek_transfor_result)

	return seg_result

def convertXmlDataToSeggedData(xml_data_file):
	'''
	xml_data: record_key	[['type':'c',;'data':'xxx'],....]
	'''
	id2data_list = {}
	id2data = {}

	for line in file(xml_data_file):
		line_list = line.strip().split('\t')

		if len(line_list) != 2:
			continue

		try:
			record_key = line_list[0]
			data_list = json.loads(line_list[1])

		except Exception,e:
			print 'Err:' + str(e)
			continue
		
		id2data[record_key] = line_list[1]
		id2data_list[record_key] = data_list

	print 'get_data from: ' + xml_data_file + '\t' + str(len(id2data_list))

	seg_result = segData(id2data_list)

	print 'seg_data from:' + xml_data_file + '\t' + str(len(seg_result))

	return id2data,seg_result

if __name__ == '__main__':
	result_file = 'result.txt'

	seg_result = convertDataFromIflytekResultFileToSeggedData(result_file)

	for cand_id in seg_result:
		print cand_id + '\t' + seg_result[cand_id]
