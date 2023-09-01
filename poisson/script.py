#runnung fno.py for different reolutions on different gpus

import subprocess
import argparse

resolutions = [32, 64, 128, 256, 512]
gpu_infos = [1, 2, 6, 7, 8] #[1, 2, 3, 4, 5]

parser = argparse.ArgumentParser(description='parse mode')
parser.add_argument('--mode' , default ='fwd', type = str, help='fwd for forward, inv for inverse')#
args = parser.parse_args()

for res, gpu_in in zip(resolutions, gpu_infos):
    if args.mode == 'inv':
        screen_name = 'inv-ufno_'+str(res)#'inv_fno_'+str(res)
        command =  'python inv-ufno.py --res %s'%(res)
    if args.mode == 'fwd':
        screen_name = 'ufno_'+str(res)#'inv_fno_'+str(res)
        command =  'python ufno.py --res %s'%(res)

    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)