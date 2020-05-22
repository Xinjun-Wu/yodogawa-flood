import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import glob
from PIL import Image
import time
import datetime

def make_image(data1, data2, outputname, supertitle ,size=(8, 24), dpi=100):

    # max_value = int(np.max((data1,data2))+1)
    max_value = 6.0
    fig,(ax0,ax1,ax2) = plt.subplots(ncols=3,figsize = (8,3))
    fig.suptitle(supertitle, fontsize='large')

    im0 = ax0.imshow(data1,cmap = cm.jet,clim=(0,max_value))
    cb0 = fig.colorbar(mappable=im0,ax=ax0)
    ax0.set_title('Input')
    ax0.axis('off')

    im1 = ax1.imshow(data2,cmap = cm.jet,clim=(0,max_value))
    cb1 = fig.colorbar(mappable=im1,ax=ax1)
    ax1.set_title('Output')
    ax1.axis('off')

    im2 = ax2.imshow(data1-data2,cmap=cm.Blues,clim = (-0.5,0.5))
    cb2 = fig.colorbar(mappable=im2,ax=ax2)
    ax2.set_title('Input-Output')
    ax2.axis('off')

    plt.tight_layout
    plt.savefig(outputname, bbox_inches='tight')
    plt.close()

def check_path(path):
    if not  os.path.exists(path):
        os.makedirs(path)


def image2gif(input_folder,outputname):
    pngList = glob.glob(input_folder + "\*.png")
    images = []
    for png in pngList:
        im=Image.open(png)
        images.append(im)
    images[0].save(input_folder+os.sep+outputname, save_all=True, append_images=images, loop=0, duration=1000)

class CurrentTime():
    def __init__(self):
        self.timestamp = []
        self.mark = []
        self.item = 0
    
    def set_point(self, mark = 'DefultName'):
        now = int(time.time())
        self.timestamp.append(now)
        print(time.asctime(now))

        if self.item != 0:
            seconds_start = self.timestamp[self.item-1]
            seconds_end = self.timestamp[self.item]
            start = datetime.timedelta(seconds=seconds_start)
            end = datetime.timedelta(seconds=seconds_end)
            timeusage = str(end - start)
            print(mark," : ",timeusage)

    
if __name__ == '__main__':
    input_folder = r'E:\Wu\Flood\20191127_計算ケース追加\save\TimeStep_3\test_results\image'
    outputname = 'test.gif'
    image2gif(input_folder,outputname)


