from glob import glob
import os
import git
# get all *.py files
depth = 5
files = []

def find_master_file(dev_file):
    cc_file = 'branch-master/ccdetection/'+dev_file.replace('branch-dev/', '')
    mm_file = 'branch-master/mmdetection/'+dev_file.replace('branch-dev/', '')
    return cc_file if os.path.exists(cc_file) else mm_file
    # return mm_file

for d in range(1, depth+1):
    exp = 'branch-dev/*/' + '*/'*d + '*.py'
    dev_files = glob(exp)
    for dev_file in dev_files:
        master_file = find_master_file(dev_file)
        if os.path.exists(master_file):
            dev_str = open(dev_file, 'r').readlines()
            master_str = open(master_file, 'r').readlines()
            if dev_str != master_str:
                print(dev_file,'->', master_file.split('/')[1])

                new_path = '../dev-detection/'+dev_file.replace('branch-dev', '')
                os.makedirs(os.path.dirname(new_path), exist_ok=1)
                os.system(f'cp {dev_file} {new_path}')
    #             import ipdb; ipdb.set_trace()
    #         # import ipdb; ipdb.set_trace()
    #     #     print(os.path.basename(dev_file))
    #     #     import ipdb; ipdb.set_trace()
    # # print('Expression:', exp, files)

