data_dir=/home/dengzhilong/work/call_reason/henan_data/2017080x_1/
data_dir=/home/dengzhilong/work/call_reason/henan_data/2017080x_2/
data_dir=/home/dengzhilong/work/call_reason/henan_data/2017080x_3/

result_dir=/home/dengzhilong/work/call_reason/henan_data/result/

for data_file in `ls $data_dir`
do
	in_file=$data_dir$data_file
	result_file=$result_dir$data_file".res"
	echo "begin ..."$data_file
	python classifyXmlDataFile.py -c henan_callreason.cfg -i $in_file -o $result_file
	
	echo "finish "$data_file
done
