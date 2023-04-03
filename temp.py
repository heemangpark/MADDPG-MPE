import os

curdir = os.getcwd()
files = list(os.walk('models'))[0][2]

for f in files:
    new = str()
    left = f.split('_')[:-1]
    right = str(int(f.split('_')[-1].split('.')[0]) + 20000) + '.pt'
    for l in left:
        new += l + '_'
    new += right
    os.rename(os.path.join(curdir, 'models', f), os.path.join(curdir, 'models', new))
