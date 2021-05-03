import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv import utils
from gluoncv.model_zoo import get_model
from os import walk
from gluoncv.utils.filesystem import try_import_decord
import pandas as pd
from os import listdir
import os
import glob
import argparse

def getModelPredictionsPerShotSlowFast(shotPath, category, model_name,logPath,trackLog):
    
    """After we extract the shots in the clips using the previous method.
    We take 32 random frame and make a prediction for each shot and then save it as a csv
    shotPath: The folder with the extracted shots
    category: sport category which we want to get predictions for
    model_name: which model from gluoncv you want to use
    logPath: Folder with all the logs
    trackLog: folder with log to track the videos that have been processed
    """
    
#     mypath_ = mypath + category+"/Shots"
    mypath_ = shotPath+category
    folders = glob.glob(os.path.join(mypath_, '*'))
    
    if not os.path.exists(trackLog+category+"_ModelPredictionCompleted.txt"):
        track_log = open(trackLog+category+"_ModelPredictionCompleted.txt", 'a')
        track_log.write("VideoName\n")
    else:
        track_log = open(trackLog+category+"_ModelPredictionCompleted.txt", 'a')
    
    logfile = open(logPath+category+"_Slowfast_log.txt", 'a')
    logfile.write(category)
    cols = ["Title", "Shot Number","Label1","Prediction Probability1","Label2","Prediction Probability2",
            "Label3","Prediction Probability3","Label4","Prediction Probability4",
            "Label5","Prediction Probability5"]
            
    old_shots = list()
    #check if shot exists in the csv file and then dont run model on shot twice
    for folder in folders:
        logfile.write(folder)
        shots = glob.glob(os.path.join(folder, '*'))
        
        df = pd.DataFrame(columns = cols)
        
        if os.path.exists(folder+"/"+"ModelPredictions.csv"):
            df_old = pd.read_csv(folder+"/"+"ModelPredictions.csv", usecols=cols)
            old_shots = df_old["Shot Number"]
        for index,shot in enumerate(shots):

            try:
                decord = try_import_decord()
                if shot.endswith(".mp4") and shot not in old_shots:
                    vr = decord.VideoReader(shot)
                    logfile.write("encoded shots")
#                     frame_id_list = range(0, 64, 2)
                    fast_frame_id_list = range(0, 64, 2)
                    slow_frame_id_list = range(0, 64, 16)
                    frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)
                    video_data = vr.get_batch(frame_id_list).asnumpy()
                    clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]

                    transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    clip_input = transform_fn(clip_input)
                    clip_input = np.stack(clip_input, axis=0)
                    clip_input = clip_input.reshape((-1,) + (36, 3, 224, 224))
                    clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))


                    net = get_model(model_name, nclass=400, pretrained=True)


                    pred = net(nd.array(clip_input))

                    classes = net.classes

                    ind = nd.topk(pred, k=5)[0].astype('int')
                    predictions = list()
                    for i in range(5):
                        predictions.append((classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))

                    #change here to match new df

                    scene = shot.rsplit('/',1)[1]
                    video_title = folder.rsplit('/',1)[1]
                    df.loc[index,cols] = [video_title,scene[:-4],predictions[0][0],predictions[0][1],predictions[1][0],predictions[1][1],
                                         predictions[2][0],predictions[2][1],predictions[3][0],predictions[3][1],
                                         predictions[4][0],predictions[4][1]]
                    predictions = list()

            except Exception as exc:
                print(str(exc))
                logfile.write(str(exc)+"\n")
                pass
            
        if os.path.exists(folder+"/"+"ModelPredictions.csv"):
            df_old = pd.read_csv(folder+"/"+"ModelPredictions.csv", usecols=cols)
            df_old.append(df,ignore_index="True")
            df_old.to_csv(folder+"/"+"ModelPredictions.csv")
            logfile.write("append successful")
            track_log.write(folder+"\n")
        else:
            df.to_csv(folder+"/"+"ModelPredictions.csv")
            logfile.write("file saved successful")
            track_log.write(folder+"\n")
    logfile.close()
    track_log.close()

def main():
	getModelPredictionsPerShotSlowFast(args.shotPath, args.category, args.model_name,args.logPath,args.trackLog)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This programs runs the classification model(slowfast) on each shot in the video ')
    parser.add_argument("--shotPath",
        help="The path to where the shots per videos are extracted",
        default = "/science-nfs/vsm01/projects/carpe-diem/ShotPerVideos/")
    parser.add_argument("--trackLog",help="The path to where we keep track of videos that has been pased to the model",default="/science-nfs/vsm01/projects/carpe-diem/preprocessing/CodeForGettingActionClips/TrackPreprocessing/")
    parser.add_argument("--category",
        help="The category we want to run prediction on, it must start with a capital letter e.g Bowling, Vault")
    parser.add_argument("--model_name",
        help="The action recognition model we want to use. Select from gluoncv",default="slowfast_4x16_resnet50_kinetics400")
    parser.add_argument("--logPath",
        help="The folder where the logs will be stored",
        default="/science-nfs/vsm01/projects/carpe-diem/preprocessing/CodeForGettingActionClips/Extracting_shots_per_video/logs/")
    args = parser.parse_args()
    
    main()














