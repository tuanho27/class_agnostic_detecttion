import os, glob
root_dir = os.path.dirname(os.path.realpath(__file__))

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def clean_backup_file(files_list):
	for f in files_list:
		new_f = f
		while 'backup' in new_f:
			new_f = f.replace('backup','')
		if new_f != f:
			os.rename(f,new_f)

ccfiles = getListOfFiles(root_dir+'/ccdetection/')

for src_file in ccfiles:
	f=src_file.replace(root_dir,'')
	print(f)
	dst_file = src_file.replace('ccdetection','mmdetection')
	dst_parent = os.path.dirname(dst_file)
	os.makedirs(dst_parent,exist_ok=True)
	if os.path.islink(dst_file):
		os.unlink(dst_file)
	if os.path.exists(dst_file) and '.py' in dst_file:
		os.rename(dst_file, dst_file.replace('.py','_backup.py'))

	if not os.path.isdir(dst_file):
		os.symlink(src_file, dst_file)

# def get_all_backup_paths(dir, level=3):
# 	list_files =[]
# 	dirs = dir + level* '/*'
# 	list_dir = [d for d in glob.glob(dirs) if os.path.isdir(d)]
# 	for dir_l in list_dir:
# 		# Get Children dir
# 		dir_c = dir_l + '/*'
# 		list_dir_c = [d for d in glob.glob(dir_c) if os.path.isdir(d)]
# 		# Get all python files from this level
# 		filenames = dir_l  + '/*.py'
# 		list_files += list_dir_c + [d for d in glob.glob(filenames) if os.path.isfile(d)]
# 		#Python files of previous levels to the root levels
# 		for l in range(level):
# 			dir_l = os.path.dirname(dir_l)
# 			filenames = dir_l  + '/*.py'
# 			list_files += [d for d in glob.glob(filenames) if os.path.isfile(d)]
# 	return list_files

# print('Create Symbolic Link for:')
# ccfiles = get_all_backup_paths(dir='ccdetection', level=3)
# root_dir = os.path.dirname(os.path.realpath(__file__))

# for f in ccfiles:
# 	print(f)
# 	src_file = os.path.join(root_dir,f)
# 	dst_file = src_file.replace('ccdetection','mmdetection')
# 	dst_parent = os.path.dirname(dst_file)
# 	os.makedirs(dst_parent,exist_ok=True)
# 	if os.path.islink(dst_file):
# 		os.unlink(dst_file)
# 	if os.path.exists(dst_file) and '.py' in dst_file:
# 		os.rename(dst_file, dst_file.replace('.py','_backup.py'))

# 	if not os.path.isdir(dst_file):
# 		os.symlink(src_file, dst_file)

