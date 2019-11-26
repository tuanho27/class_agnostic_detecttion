from glob import glob
import os
os.remove('_mm_status.txt')
os.system('cd mmdetection && echo $(git status) >> ../_mm_status.txt')

assert os.path.exists('origin_mmdetection'), 'make sure you have a copy of mmdetection, take a look at install.sh'

# os.system(
#     'cp -r mmdetection origin_mmdetection && cd origin_mmdetection && git checkout -- .')


#Step1 Copy new, modified files
def find_all_file(text):
    possible_paths = text.split(' ')
    paths = []
    for path in possible_paths:
        if os.path.exists('mmdetection/'+path) and not '_backup.py' in path:
            paths.append(path)
    return paths

text = open('_mm_status.txt', 'r').readline()
paths = find_all_file(text)
for path in paths:
    not_symlink = not os.path.islink(f'mmdetection/{path}')
    is_none_existed_dir = os.path.isdir(
        f'mmdetection/{path}') and not os.path.exists(f'ccdetection/{path}')
    is_file = os.path.isfile(f'mmdetection/{path}')

    if not_symlink and (is_none_existed_dir or is_file):
        print(f'copy-to-cc: {path} ')
        os.system(f'cp -r mmdetection/{path} ccdetection/{path} ')

##Step2 remove duplicate file with origin mm
def is_same_content(a, b):
    try:
        if open(a).readlines() == open(b).readlines():
            return True
        return False
    except:
        return False


cc_files = []

for lvl in range(6):
    glob_qr = 'ccdetection/'+'*/'*lvl+'*.py'
    cc_files += glob(glob_qr)


for cc_file in cc_files:
    origin_mmfile = cc_file.replace('ccdetection/', 'origin_mmdetection/')
    if is_same_content(cc_file, origin_mmfile):
        os.remove(cc_file)
        print('remove for being duplicate with origin mmdetection:', cc_file)

os.system('rm -rf origin_mmdetection')


