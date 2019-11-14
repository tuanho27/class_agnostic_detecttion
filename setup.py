import os, glob

def get_all_backup_paths(dir, level=3):
	list_files =[]
	dirs = dir + level* '/*'
	list_dir = [d for d in glob.glob(dirs) if os.path.isdir(d)]
	for dir_l in list_dir:
		# Get Children dir
		dir_c = dir_l + '/*'
		list_dir_c = [d for d in glob.glob(dir_c) if os.path.isdir(d)]
		# Get all python files from this level 
		filenames = dir_l  + '/*.py'
		list_files += list_dir_c + [d for d in glob.glob(filenames) if os.path.isfile(d)]
		#Python files of previous levels to the root levels
		for l in range(level):
			dir_l = os.path.dirname(dir_l)
			filenames = dir_l  + '/*.py'
			list_files += [d for d in glob.glob(filenames) if os.path.isfile(d)]
	return list_files

ccfiles = get_all_backup_paths(dir='ccdetection', level=3)

root_dir = os.path.dirname(os.path.realpath(__file__))
print(root_dir)
print('Create Symbolic Link for:')
for f in ccfiles:
	print(f)
	src_file = os.path.join(root_dir,f)
	dst_file = src_file.replace('ccdetection','mmdetection') 
	dst_parent = os.path.dirname(dst_file)
	os.makedirs(dst_parent,exist_ok=True)
	if os.path.exists(dst_file):
		if os.path.islink(dst_file):
			os.unlink(dst_file)
		elif '.py' in dst_file:
			os.rename(dst_file, dst_file.replace('.py','backup.py'))
	os.symlink(src_file, dst_file)

