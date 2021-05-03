import pandas as pd
import glob
import os
import numpy as np
from moviepy.editor import *
import numpy as np
import argparse

def rankingForRankWithLikelihood(shot_num,prediction_positions):
    """This method is part of the bigger algorithm
        It takes a list containing the shots that contains the category and the rank/likelihood the model gave to it.
        Here we get a list of the shots in decending order so the shot that has the required category predicted with the lowest likelihood comes first
        In our case the lowest likelihoods appears in the fifth label at highest appears in label 1
        
        """

    tmp = prediction_positions
    rank_shot = []
    for i in range(len(tmp)):
#         if len(rank_shot) <= len(prediction_positions):
        max_val = max(tmp)
        max_indx = tmp.index(max_val)

        rank_shot.append(shot_num[max_indx])
        tmp[max_indx] = -1
            
    return list(set(rank_shot))




            
def getClipDuration(path,shot):
    clip = VideoFileClip(path+str(shot)+".mp4")
    duration = int(clip.duration)
    duration %= 3600
    return duration


def makeAnnotationClipKeepOrder(shots_ranked,category,videos,shotPath):
    """This function creates a video clip while taking note of temporal order"""
    
    shots_ = []
    shots_length = []
    path = shotPath+category+"/"+videos+"/"
    clip_length = 0
    final_shot = []
    init = shots_ranked[0]
    dur = getClipDuration(path,init)
    shots_length.append(dur)
    shots_.append(init)
    shots_ranked_ = np.array(shots_ranked)
    for indx in range(1,len(shots_ranked)):
        
        if sum(shots_length) > 60:
            print("length reached")
            break
        else:
            before = min(shots_)-1
            after = max(shots_)+1
            if min(shots_ranked) <= before and max(shots_ranked) >= after:
                # print("first if")
                # print("before ",before," after ",after)
                closest_min_val_in_rank = shots_ranked_[shots_ranked_ <= before].max()
                closest_max_val_in_rank = shots_ranked_[shots_ranked_ >= after].min()
                print(closest_min_val_in_rank,closest_max_val_in_rank)
                if shots_ranked.index(closest_min_val_in_rank) < shots_ranked.index(closest_max_val_in_rank):
                    
                    dur = getClipDuration(path,closest_min_val_in_rank)
                    
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60")
                    else:
                        shots_.insert(shots_.index(min(shots_)),closest_min_val_in_rank)

                        shots_length.append(dur)
                else:
                    dur = getClipDuration(path,closest_max_val_in_rank)
                    
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60\n")
                    else:
                        shots_.insert(shots_.index(min(shots_))+1,closest_max_val_in_rank)
                        shots_length.append(dur)
            elif min(shots_ranked) > before and max(shots_ranked) >= after:
                # print("second if")
                before = max(shots_)+1
                after = max(shots_ranked)
                # print("before ",before," after ",after)
                closest_min_val_in_rank = shots_ranked_[shots_ranked_ >= before].min()
                closest_max_val_in_rank = shots_ranked_[shots_ranked_ >= after].min()
                # print(closest_min_val_in_rank,closest_max_val_in_rank)
                if shots_ranked.index(closest_min_val_in_rank) < shots_ranked.index(closest_max_val_in_rank):
                    
                    dur = getClipDuration(path,closest_min_val_in_rank)
                    
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60\n")
                    else:
                        shots_.insert(shots_.index(max(shots_))+1,closest_min_val_in_rank)
                    
                        shots_length.append(dur)
                else:
                    dur = getClipDuration(path,closest_max_val_in_rank)
                    
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60\n")
                    else:
                        shots_.insert(shots_.index(max(shots_))+1,closest_max_val_in_rank)
                        shots_length.append(dur)
            elif min(shots_ranked) <= before and max(shots_ranked) < after:
                # print("3rd if")
                before = min(shots_)-1
                after = min(shots_ranked)
                # print("before ",before," after ",after)
                closest_min_val_in_rank = shots_ranked_[shots_ranked_ <= before].max()
                closest_max_val_in_rank = shots_ranked_[shots_ranked_ <= after].max()
                # print(closest_min_val_in_rank,closest_max_val_in_rank)
                if shots_ranked.index(closest_min_val_in_rank) < shots_ranked.index(closest_max_val_in_rank):
                    dur = getClipDuration(path,closest_min_val_in_rank)
                    
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60\n")
                    else:
                        shots_.insert(shots_.index(min(shots_)),closest_min_val_in_rank)                    
                        shots_length.append(dur)
                else:
                    dur = getClipDuration(path,closest_max_val_in_rank)
                    if (sum(shots_length) + dur) > 60:
                        print("clip now more than 60\n")
                    else:
                        shots_.insert(shots_.index(min(shots_)),closest_max_val_in_rank)
                        shots_length.append(dur)
    #form the clip
    for shot in shots_:
        clip = VideoFileClip(path+str(shot)+".mp4")
        final_shot.append(clip)
    return list(set(final_shot))
              
def rankKeepOrder(category,shotPath,savePath2,logPath,trackLog):

    
    """Similar algorithm as above but we re-rank the ranked shot to keep some temporal order"""
    
    path = shotPath+category+"/"
    video_folders = os.listdir(path)
#     print(video_folders)
    cols = ["Label1","Label2","Label3","Label4","Label5"]
    unusedVideos = []
    if not os.path.exists(trackLog+category+"_RankingCompleted.txt"):
        track_log = open(trackLog+category+"_RankingCompleted.txt", 'a')
        track_log.write("VideoName\n")
    else:
        track_log = open(trackLog+category+"_RankingCompleted.txt", 'a')
        
    logfile = open(logPath+category+"_Rankinglog.txt", 'a')
    
    multi_value_categories = ["Baseball","Basketball","Softball","Golfing","Soccer","American football","Kickball","Tennis","Badminton","Volleyball","Frisbee","Cricket"]
    baseball = ["catching_or_throwing_baseball", "hitting_baseball"]
    basketball = ["dribbling_basketball","dunking_basketball","playing_basketball","shooting_basketball"]
    softball = ["catching_or_throwing_softball"]
    golfing = ["golf_chipping","golf_driving","golf_putting"]
    soccer = ["shooting_goal_ -soccer-", "juggling_soccer_ball", "kicking_soccer_ball"]
    american_football = ["passing_American_football_-in_game-", "passing_American_football_-not_in_game-","kicking_field_goal"]
    kickball = ["playing_kickball"]
    tennis = ["playing_tennis"]
    badminton = ["playing_badminton"]
    volleyball = ["playing_volleyball"]
    frisbee = ["catching_or_throwing_frisbee"]
    cricket = ["playing_cricket"]
    save_path = savePath2+category+"/"
    
    for videos in video_folders:
        print(videos)
        if not videos.startswith(".") and not os.path.exists(save_path+videos+".mp4"):#check that we dont use hidden file and the clip hasnt already been extracted
        
            predictions = pd.read_csv(path+videos+"/ModelPredictions.csv")
            prediction_positions = []
            shot_num = []
            shot_numbers = predictions["Shot Number"]
            for shot,shot_val in enumerate(shot_numbers):

                predicted_values = list(predictions.loc[shot,cols])
                category_ = category.lower()
                
                if category not in multi_value_categories:
                    if category_.replace(" ","_") in predicted_values:
                        shot_num.append(shot_val)
                        position_num = predicted_values.index(category_.replace(" ","_")) + 1 #+1 to make it start at 1 not zero so we have 1-5 not 0-4
                        prediction_positions.append(position_num)
                else:
                    if category == "Baseball":
                        if (baseball[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(baseball[0]) + 1
                            prediction_positions.append(position_num)
                        elif  (baseball[1] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(baseball[1]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Basketball":
                        if (basketball[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(basketball[0]) + 1
                            prediction_positions.append(position_num)
                        elif (basketball[1] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(basketball[1]) + 1
                            prediction_positions.append(position_num)
                        elif (basketball[2] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(basketball[2]) + 1
                            prediction_positions.append(position_num)
                        elif (basketball[3] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(basketball[3]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Softball":
                        if (softball[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(softball[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Frisbee":
                        if (frisbee[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(frisbee[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Cricket":
                        if (cricket[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(cricket[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Kickball":
                        if (kickball[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(kickball[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Tennis":
                        if (tennis[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(tennis[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Badminton":
                        if (badminton[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(badminton[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Volleyball":
                        if (volleyball[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(volleyball[0]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Golfing":
                        if (golfing[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(golfing[0]) + 1
                            prediction_positions.append(position_num)
                        elif (golfing[1] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(golfing[1]) + 1
                            prediction_positions.append(position_num)
                        elif (golfing[2] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(golfing[2]) + 1
                            prediction_positions.append(position_num)
                    elif category == "Soccer":
                        if (soccer[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(soccer[0]) + 1
                            prediction_positions.append(position_num)
                        elif (soccer[1] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(soccer[1]) + 1
                            prediction_positions.append(position_num)
                        elif (soccer[2] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(soccer[2]) + 1
                            prediction_positions.append(position_num)
                    elif category == "American football":
                        if (american_football[0] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(american_football[0]) + 1
                            prediction_positions.append(position_num)
                        elif (american_football[1] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(american_football[1]) + 1
                            prediction_positions.append(position_num)
                        elif (american_football[2] in predicted_values):
                            shot_num.append(shot_val)
                            position_num = predicted_values.index(american_football[2]) + 1
                            prediction_positions.append(position_num)
            
            
            
            
            print(shot_num,prediction_positions)
            if(len(prediction_positions)==0):
                unusedVideos.append(videos)
                track_log.write("unused video "+videos+"\n")
            else:
                shots_ranked = rankingForRankWithLikelihood(shot_num,prediction_positions)
                shot_s = makeAnnotationClipKeepOrder(shots_ranked,category,videos,shotPath)
#                 print(shots_ranked,shot_s)
                
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists("/science-nfs/vsm01/projects/carpe-diem/clipLessThan5/"+category):
                    os.mkdir("/science-nfs/vsm01/projects/carpe-diem/clipLessThan5/"+category)
                try:
                    final_clip = concatenate_videoclips(shot_s)
                    duration_final = int(final_clip.duration)
                    save_fivesec = "/science-nfs/vsm01/projects/carpe-diem/clipLessThan5/"+category+"/"
                    if duration_final < 5:
                        print("trying to write less than 5 sec clip \n")
                        final_clip.write_videofile(save_fivesec+videos+".mp4", temp_audiofile=save_fivesec+'temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
                        print("file saved: clip less than 5! \n")
                        track_log.write(videos+" saved clip less than 5! \n")
                    else:
                        print("trying to write more than 5 sec clip \n")
                        final_clip.write_videofile(save_path+videos+".mp4", temp_audiofile=save_path+'temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
                        print("file saved!")
                        track_log.write(videos+" saved!\n")
                except Exception as exc:
                    logfile.write(str(exc)+"\n")

def main():
#    print("First rank")
#    rankWithLikelihood(args.category,args.shotPath,args.savePath1,args.logPath)
    print("Second rank")
    rankKeepOrder(args.category,args.shotPath,args.savePath2,args.logPath,args.trackLog)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This programs runs the the 2 variation of ranking algorithm ')
    parser.add_argument("--shotPath",
         help="The path to where the shots per videos are extracted",
         default = "/science-nfs/vsm01/projects/carpe-diem/ShotPerVideos/")
    parser.add_argument("--category",
         help="The category we want to run prediction on, it must start with a capital letter e.g Bowling, Vault")
    parser.add_argument("--savePath1",
         help="The path to where we want to save the final clip generated by ranking algorithm 1",
         default="/science-nfs/vsm01/projects/carpe-diem/AnnoationSampleClip1/")
    parser.add_argument("--savePath2",
         help="The path to where we want to save the final clip generated by ranking algorithm 2",
         default="/science-nfs/vsm01/projects/carpe-diem/AnnotationSampleClip2/")
    parser.add_argument("--logPath",
         help="The folder where the logs will be stored",
         default="/science-nfs/vsm01/projects/carpe-diem/preprocessing/CodeForGettingActionClips/logs/")
    parser.add_argument("--trackLog",
         help="The path to where we keep track of videos that has been pased to the model",default="/science-nfs/vsm01/projects/carpe-diem/preprocessing/CodeForGettingActionClips/TrackPreprocessing/")
    
    args = parser.parse_args()
    main()
