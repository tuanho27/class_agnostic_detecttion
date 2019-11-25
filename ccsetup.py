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
		while '_backup' in new_f:
			new_f = f.replace('_backup','')
		if new_f != f:
			os.rename(f,new_f)

ccfiles = getListOfFiles(root_dir+'/ccdetection/')

for src_file in ccfiles:
	f=src_file.replace(root_dir,'')
	dst_file = src_file.replace('ccdetection','mmdetection')
	dst_parent = os.path.dirname(dst_file)
	os.makedirs(dst_parent,exist_ok=True)
	if os.path.islink(dst_file):
		os.unlink(dst_file)
	if os.path.exists(dst_file) and '.py' in dst_file:
		os.rename(dst_file, dst_file.replace('.py','_backup.py'))

	if not os.path.isdir(dst_file):
		print(f)
		os.symlink(src_file, dst_file)
