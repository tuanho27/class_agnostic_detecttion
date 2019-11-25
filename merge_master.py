from glob import glob
import os
import git
# get all *.py files
depth = 2
files = []

def find_master_file(dev_file):
    cc_file = 'ccdetection/'+dev_file
    mm_file = 'mmdetection/'+dev_file
    return cc_file if os.path.exists(cc_file) else mm_file

for d in range(1, depth+1):
    exp = 'mmdet/' + '*/'*d + '*.py'
    dev_files = glob(exp)
    for dev_file in dev_files:
        master_file = find_master_file(dev_file)
        print(git.diff(dev_file, master_file))
        import ipdb; ipdb.set_trace()
    # print('Expression:', exp, files)
