import numpy as np
from os import walk
import pandas as pd
from os import listdir
from gluoncv.utils.filesystem import try_import_decord
import os
import glob
from moviepy.editor import *
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse



def extractShots(pathTovideo,shotPaths,scenesFile,category,logPath):
    """We extract the shots from each video. Basically we get multiple video clips based on the
    number of shots we have in the video
        mypath:mount point of carpe-diem drive
        scenes:scenes/scences.csv contating result of shot detection
        category:The category you want to get shots of
    """
    scenes_ = pd.read_csv(scenesFile)
    scene = scenes_[scenes_["Category"] == category]
    mypath_ = pathTovideo + category #path to videos
    videos = scene["Video"]
    start_time = list(scene["Start Time (seconds)"])
    end_time = list(scene["End Time (seconds)"])
    scene_number = list(scene["Scene Number"])
    logfile = open(logPath+category+"_ExtractionTrack.txt", 'a')
    logfile.write("Videoname \n")
    for index,video_ in enumerate(videos):

        dir_ = shotPaths+category+"/"
        try:
#            videoName_nospace = video_[:-4].replace(" ","_")
            output_path = dir_ + video_[:-4] +"/"
#            file_path = dir_+videoName_nospace+""

            if not os.path.exists(dir_):#make path for category within the shotperCategory follder
                os.mkdir(dir_)
                print("directory created")

            if not os.path.exists(output_path): #creates the video folde
                os.mkdir(output_path)
                print("file created")
                
#             if os.path.exists(output_path+str(scene_number[index])+".mp4"):
                
            if os.path.exists(output_path+str(scene_number[index])+".mp4"):
                print("extraction done before")
            else:
                #we only want to extract when we havent extracted before a.ka when the video folder doesnt exist within thecategory folder
                # decord = try_import_decord()
                print("Extracting.....")

#                 ffmpeg_extract_subclip(mypath_ +"/"+video_, start_time[index], end_time[index],targetname= output_path+str(scene_number[index])+".mp4")
                logfile.write("error here 1: "+mypath_ +"/"+video_)
                clip = VideoFileClip(mypath_ +"/"+video_)
                logfile.write("\n error here 2")
                clip = clip.subclip(start_time[index], end_time[index])
                logfile.write("\n About to extract \n")
                clip.write_videofile(output_path+str(scene_number[index])+'.mp4',temp_audiofile=output_path+'temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
                logfile.write(output_path+str(scene_number[index])+".mp4\n")
    

        except Exception as exc:
            logfile.write(str(exc)+"\n")
            
    logfile.close()

def main():
  extractShots(args.pathTovideo,args.shotPaths,args.scenesFile,args.category,args.logPath)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This programs extracts the shots in each video.This is useful for making the final annotation clips')
  parser.add_argument("--pathTovideo",help="The path to where the videos for all categories is saved",default = "/science-nfs/vsm01/projects/carpe-diem/videos/")
  parser.add_argument("--shotPaths",help="The folder where the extracted clips will be stored",default="/science-nfs/vsm01/projects/carpe-diem/ShotPerVideos/")
  parser.add_argument("--scenesFile",help="The path to where scenes(now called child scenes) is saved",default="/science-nfs/vsm01/projects/carpe-diem/preprocessed-data/child_scenes.csv")
  parser.add_argument("--category",help="The category we want to run extraction on, it must start with a capital letter e.g Bowling, Vault")
  parser.add_argument("--logPath",help="The folder where the logs will be stored",default="/science-nfs/vsm01/projects/carpe-diem/preprocessing/CodeForGettingActionClips/logs/")
  args = parser.parse_args()
  main()



