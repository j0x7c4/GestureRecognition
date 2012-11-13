import sys
import os

def genTrainList (root_path,object_name):
	dir_list = os.listdir(root_path)
	file_handle = open(object_name+"_svm_data","w")
	for dir_name in dir_list:
		catg = '0'
		if ( dir_name == object_name ):
			catg = '1'
		for image_name in os.listdir(root_path+"/"+dir_name):
			if ( image_name.endswith(".jpg") or image_name.endswith(".png") ):
				file_handle.write(catg+"\n"+root_path+"/"+dir_name+"/"+image_name+"\n")

	file_handle.close()



# --- Main --- #
if __name__ == '__main__':
	if len(sys.argv) > 2:
		root_path = sys.argv[1]
		object_name = sys.argv[2]
		genTrainList(root_path,object_name)
	else:
		print "####Usage####"
		print "argv[1]: root path"
		print "argv[2]: object name"