import os
os.remove('_mm_status.txt')
os.system('cd mmdetection && echo $(git status) >> ../_mm_status.txt')
os.system('cp -r mmdetection origin_mmdetection && cd origin_mmdetection && git checkout -- .')

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
    # import ipdb; ipdb.set_trace()
    not_symlink = not os.path.islink(f'mmdetection/{path}')
    is_none_existed_dir = os.path.isdir(f'mmdetection/{path}') and  not os.path.exists(f'ccdetection/{path}')
    is_file = os.path.isfile(f'mmdetection/{path}')

    if not_symlink and (is_none_existed_dir or is_file):
        print(f'cp -r mmdetection/{path} ccdetection/{path} ')
        os.system(f'cp -r mmdetection/{path} ccdetection/{path} ')
        

        
# os.remove('_mm_status.txt')
# os.system('cp -r mmdetection origin_mmdetection && cd origin_mmdetection && git checkout -- .')
# os.system('echo $(git status) >> _mm_status.txt')


# text = open('_mm_status.txt', 'r').readline()
# possible_paths = text.split(' ')
# for path in possible_paths:
#     if os.path.exists(path) and  'ccdetection/' in path:
#         str_1 = open(path).readlines()
#         origin_path = path.replace('ccdetection', 'origin_mmdetection')
#         if os.path.exists(origin_path):
#             str_2 = open(origin_path).readlines()
#             if str_1 == str_2:
#                 print('remove for being duplicate with origin mmdetection:', path)


os.system('rm -r origin_mmdetection')