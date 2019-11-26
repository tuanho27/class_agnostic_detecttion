import os

os.system('cd mmdetection && echo $(git status) >> ../_mm_status.txt')

def find_all_file(text):
    possible_paths = text.split(' ')
    paths = []
    for path in possible_paths:
        print(path)
        if os.path.exists('mmdetection/'+path):
            paths.append(path)
    return paths 

text = open('_mm_status.txt', 'r').readline()
paths = find_all_file(text)
for path in paths:
    print(f'cp -r mmdetection/{path} ccdetection/{path} ')
    