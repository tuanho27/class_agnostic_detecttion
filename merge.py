import os

# os.system(' git reset master --hard   ')
os.system(' git checkout dev ccdetection   ')

if os.path.exists('gitdiff.txt'):
    os.remove('gitdiff.txt')
os.system('echo $(git status) >> ../.gitdiff.txt')

possible_paths = open('../.gitdiff.txt', 'r').readline().split(' ')

for dev_path in possible_paths:
    if os.path.exists(dev_path)  and 'ccdetection' in dev_path :
        master_path = dev_path.replace('ccdetection', 'mmdetection')
        if os.path.isfile(master_path):        
            dev_str = open(dev_path, 'r').readlines()
            master_str = open(master_path, 'r').readlines()
            if dev_str == master_str:
                print('[Remove]', dev_path)
                os.remove(dev_path) 
                os.system('git rm '+dev_path+' --cached')
                dir_path = os.path.dirname(dev_path)
                if len(os.listdir(dir_path)) == 0:
                    print('[Remove dir]', dir_path)
                    os.system(f'rm -r {dir_path}')
