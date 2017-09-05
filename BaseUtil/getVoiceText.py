#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-07-26 11:55:23
'''
 
import sys
import os
import json


def loadDataFromIflytekResultFile(iflytek_result_file):
	'''
	load data from iflytek_result file and return data transformed data
	return: {id:[{'type':'c','data':'xxxxx'}]}
	'''
	
	iflytek_result = {}

	line_num = 0
	for line in file(result_file):
		line_num += 1
		line = line.strip()
		if line == '流水号$路径$时长$格式$文本$时间$声道$文本长度$静音$有效时长':
			continue

		line_list = line.split('$')

		if len(line_list) != 10:
			print 'Error line ' + result_file + ' ' + str(line_num)
			continue

		case_id = line_list[0]

		word_list = line_list[4].strip().split(';')

		voice_id_list = line_list[6].strip().split(';')

		if len(word_list) != len(voice_id_list):
			print 'data_cnt_error:' +  + result_file + ' ' + str(line_num)
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

			iflytek_result[case_id] = iflytek_result 
		
	print 'load result data from ' + result_file + ' get cases: ' + str(len(iflytek_result)))
	return iflytek_result


