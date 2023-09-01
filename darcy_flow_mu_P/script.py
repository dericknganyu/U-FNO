#runnung fno.py for different reolutions on different gpus

import subprocess

resolutions = [512]#[16, 32, 64, 128]#, 256, 512]
batch_size =  [5]#[10, 10, 10, 10 ]#, 10, 5]
gpu_infos =   [0]#[ 0,  0,  1,  2 ]#, 3, 4]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'ufno-darcyPWC_'+str(res)#'inv_fno_'+str(res)
    command =  'python ufno.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)

gpu_infos =   [1]#[0 , 0 , 1 , 7 ]#  , 6  , 5]

for res, gpu_in, bs  in zip(resolutions, gpu_infos, batch_size):

    screen_name = 'inv-ufno-darcyPWC_'+str(res)#'inv_fno_'+str(res)
    command =  'python inv-ufno.py --res %s --bs %s'%(res, bs)
    subprocess.run('screen -dmS '+screen_name, shell=True)
    
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'conda activate base'), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, 'export CUDA_VISIBLE_DEVICES=%s'%(gpu_in)), shell=True)
    subprocess.run('screen -r  %s -p 0 -X stuff "%s^M"'%(screen_name, command), shell=True)