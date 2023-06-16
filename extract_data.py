import os
import cv2
import re
import numpy as np
import pandas as pd
from skimage import data, io
#TODO Follow Google Cloud Vision instructions provided in read me.... os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "/PATH/TO/JSON/CONTAINING/API/KEY/FOR/GOOGLECV" <- may not be best practice. Refer to GCV page for more info.

def read_screenshots(scoreboard_path='scoreboard.png', summary_path=None):
    stats_dict = {'IGN': [], 'ACS': [], 'K': [], 'D': [], 'A': [], 'KD': [], 'KDA': [], 'ER': [], 'FB': [], 'PLANTS': [], 'DEFUSES': [], 'MAP': [], 'SID': []} #creates dictionary to add parsed GCV output into Shape(10,13)
    try:
        scoreboard = cv2.imread(scoreboard_path) # read in screenshot from cv2
    except:
        print(f'No file was able to be opened. Outputting scoreboard_path. Default is current folder to a file scoreboard.png {scoreboard_path}')
    if summary_path: 
        summary = cv2.imread(summary_path) # read in summary screen via cv2
        extract_screenshots(scoreboard, summary) # call to extract_screenshots to save cropped and upscaled images for text detection through google cloud vision
    else:
        extract_screenshots(scoreboard) # call to extract_screenshots to save cropped and upscaled images for text detection through google cloud vision

    parse_gcv_output(stats_dict) # calls parser function to get ouput from GCV and parse it

def extract_screenshots(screenshot, summary=None, KDA_SCALE=240, IGN_SCALE=500, SCALE_PERCENT=150, MISC_SCALE=300):
    #upscale images by 150% currently is what seems best for screenshots of 1920x1080
    stats = ['ACS', 'KDA', 'ER', 'FB', 'PLANTS', 'DEFUSES'] #list of scoreboard stats of same shape to create dataframe from
    IGN_XY= [[345, 390], [330, 630]] #maximum ign length based on 1920x1080 resolution 330, 345 px and 630, 390
    IGN_WIDTH = 52 # sets width of Row of IGNs
    MISC_XY = [[90, 160], [70, 275]] #location of misc information such as date and map based on 1920x1080 resolution
    STAT_COL_XY = [[345, 860], [680, 825]] # location of columns of stats such as ACS KDA ER etc...
    if summary.all():        
        SUMMARY_XY = [[350, 1300], [940, 1635]] # = [[940, 350], [1635, 865]] are x1,y1 and x2,y2 on paint respectively. (top left corner and bottom right corner) in pixels. 
        SUMMARY_IMG = summary[SUMMARY_XY[0][0]:SUMMARY_XY[1][0], SUMMARY_XY[0][1]:SUMMARY_XY[1][1]] # crops summary image based on 1920x1080
        resize_and_save(SUMMARY_IMG, SCALE_PERCENT, 'SUMMARYCROPPED') # sends summary to be upscaled and saved
    for ign_number in range(10):
        NAME_ROW_SHIFT = IGN_WIDTH * ign_number #determines how far to shift down to grab next ign
        IGN = screenshot[IGN_XY[0][0]+NAME_ROW_SHIFT:IGN_XY[0][1]+NAME_ROW_SHIFT, IGN_XY[1][0]:IGN_XY[1][1]] #crop IGN information from image
        resize_and_save(IGN, IGN_SCALE, 'IGN'+str(ign_number)) #crop upscale and save ign 

    MISC = screenshot[MISC_XY[0][0]:MISC_XY[0][1], MISC_XY[1][0]:MISC_XY[1][1]] #crop misc information from image
    
    resize_and_save(IGN, SCALE_PERCENT, 'IGN') #upscale and save cropped image of IGNs
    resize_and_save(MISC, MISC_SCALE, 'MISC') # call helper function upscale image to aid google vision in detecting small characters

    STAT_COL_NUM = 0 # set current index of columns of stats to 0
    for stat in stats: # iterate over list of stats in the scoreboard
        STAT_COL_WIDTH = 150 # width of stats in these columns is 150 pixels
        STAT_COL_SHIFT = (STAT_COL_WIDTH*STAT_COL_NUM) # calculates shift based on current index
        temp_image = screenshot[STAT_COL_XY[0][0]:STAT_COL_XY[0][1], STAT_COL_XY[1][0]+STAT_COL_SHIFT:STAT_COL_XY[1][1]+STAT_COL_SHIFT] #crops image to column of stats 
        if stat == 'KDA': # special circumstance for KDA column. Backslashes are inconsistently output from GCV
            resize_and_save(temp_image, KDA_SCALE, stat) # requires extra resolution
        else: # can proceed as normal
            resize_and_save(temp_image, SCALE_PERCENT, stat) #sends current column of stats to be upscaled and saved
        STAT_COL_NUM += 1 # increment column of stats to correctly grab the next

def resize_and_save(image, SCALE_PERCENT, stat):
    width = int(image.shape[1] * SCALE_PERCENT / 100) # set width based on SCALE_PERCENT
    height = int(image.shape[0] * SCALE_PERCENT / 100) # set height based on SCALE_PERCENT
    dim = (width, height) # set dimensions
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize image
    io.imsave(stat+'.png', image) # saves image in the form of STATNAME . png

def parse_gcv_output(stats_dict):

    MISC = parse_gcv_misc('MISC.png')

    SID = parse_gcv_sid(MISC) # creates the scrim id from MISC info
    MAP = MISC[3] # grabs which map it is from MISC GCV output
    
    stats_dict['IGN'] = parse_gcv_ign('IGN') # adds list of IGNs to dict

    SB = parse_gcv_stat_col(MAP, SID, stats_dict) # scoreboard as a dataframe
    SM = parse_gcv_summary(MAP, SID)   

    SM.to_csv(SID+'SM.csv') # saves summary dataframe as csv
    SB.to_csv(SID+'SB.csv') # saves scoreboard dataframe as csv

def parse_gcv_misc(misc_image)->list:
    MAPS = ['ASCENT', 'BREEZE', 'BIND', 'HAVEN', 'ICEBOX', 'FRACTURE', 'PEARL', 'SPLIT']
    UNWANTED = [',', '-', 'MAP', 'CUSTOM', 'STANDARD', '—'] # list of unwanted characters GCV outputs when presented with MISC crop

    MISC = google_cloud_vision(misc_image)[1:] # creates variable containing MISC information such as map date duration etc...  Example [MM, DD, YYYY, ',', 'STANDARD', 'COMPETITIVE', 'BREEZE' ]

    MISC = [i for i in MISC if i not in UNWANTED] # list comprehension removes instances of unwanted from MISC

    assert MISC[3] in MAPS, f"Map output from GCV is not correct got: {MISC[3]}"
    return MISC

def parse_gcv_sid(MISC):
    SID_regex = r"^[A-Z]+[0-9]+[A-Z]+[0-9]{4}" # matches any amount of starting numbers for MONTHDDYY - valorant may output MONTHMDYYYY

    SID = MISC[0] + MISC[1] + MISC[2] + MISC[3] + MISC[4][0] + MISC[4][1] + MISC[4][3] + MISC[4][4] # scrim id (scrim id is MONTHDDYYYYMAPDURATION) example: OCT302022BREEZE2356

    if bool(re.match(SID_regex, SID)): # make sure that SID is correct format (not a huge deal just making sure GCV output is as expected)
        print(f'Scrim ID or SID does not match regex created: SID should be MONTHDDYYMAPDURATION example: OCT302022BREEZE2356. Got: {SID}')
    return SID

def parse_gcv_ign(ign_image_prefix)->list:
    IGN = [] # creating empty list to add IGNs to before adding to dict

    for ign_number in range(10): # Iterates 10 times for 10 players and 10 cropped images.
        temp_ign_array = google_cloud_vision(ign_image_prefix+str(ign_number)+'.png')[1:] # process GCV for each ROW of IGNs (10)
        IGN.append(''.join(temp_ign_array)) # joins GCV output together (names with spaces were counting as two names)
    assert len(IGN) == 10, f'Parsed more or less than 10 IGNs. Will cause an error later when casting into DataFrame. Configure IGN_SCALE or tweak IGN parser. IGN list : {IGN}'
    return IGN

def google_cloud_vision(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io

    output = []
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        output.append(text.description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return output

def parse_gcv_summary(MAP, SID)->list:
    SUMMARY_STATS = google_cloud_vision('SUMMARYCROPPED.png')[1:] # send summary screen to GCV 
    summary_dict = {'MAP': [SUMMARY_STATS[0], SUMMARY_STATS[0]+' WINS',  SUMMARY_STATS[0]+' FB',  SUMMARY_STATS[0]+' FB WINS',  SUMMARY_STATS[0]+' ELIM WINS', 
                            SUMMARY_STATS[0]+' SPIKES DEP', SUMMARY_STATS[0]+' POST WINS', SUMMARY_STATS[0]+' DEFUSALS', SUMMARY_STATS[0]+' DETONATIONS', 
                            SUMMARY_STATS[12], SUMMARY_STATS[12]+' WINS',  SUMMARY_STATS[12]+' FB',  SUMMARY_STATS[12]+' FB WINS',  SUMMARY_STATS[12]+' ELIM WINS', 
                            SUMMARY_STATS[12]+' SPIKES DEP', SUMMARY_STATS[12]+' POST WINS', SUMMARY_STATS[12]+' DEFUSALS', SUMMARY_STATS[12]+' DETONATIONS','SID'],
                    MAP: [SUMMARY_STATS[1], SUMMARY_STATS[3],SUMMARY_STATS[4],SUMMARY_STATS[5],SUMMARY_STATS[6],SUMMARY_STATS[7],SUMMARY_STATS[8],SUMMARY_STATS[9],SUMMARY_STATS[11], 
                          SUMMARY_STATS[13], SUMMARY_STATS[15],SUMMARY_STATS[16],SUMMARY_STATS[17],SUMMARY_STATS[18],SUMMARY_STATS[19],SUMMARY_STATS[20],SUMMARY_STATS[21],SUMMARY_STATS[-1], SID]} # ugly dictionary. 
    try:
        SM = pd.DataFrame.from_dict(summary_dict) # creating dataframe from dict
    except:
            print(f'Shape is not correct, cannot cast into DataFrame. GCV likely output unexpected character. Check output and configure upscale factor. Scoreboard dict : {summary_dict}')

    SM = SM.transpose() # transposes the summary dataframe. Is more functional this way, don't want to fix initialization
    
    return SM # dataframe from gcv output of summary screenshot

def parse_gcv_stat_col(MAP, SID, stats_dict)->dict:    
    stats = ['ACS', 'KDA', 'ER', 'FB', 'PLANTS', 'DEFUSES'] # list of scoreboard stats of same shape to create dataframe from

    for stat in stats:
        if stat == 'KDA': # need especial processing to remove slashes from GCV output
            gcv_output = google_cloud_vision(stat+'.png')[1:] # process GCV on KDA column
            kda_no_slash = [i for i in gcv_output if i != '/'] # removes slashes from kda (GCV was inconsistent at detecting so remove all cases)
            assert len(kda_no_slash) == 30, f'KDA not parsed correctly. Configure KDA_SCALE or modify KDA output parser. KDA list : {kda_no_slash}, len(kda_no_slash): {len(kda_no_slash)}'
            #kills would be at index 0, 3, 6, 9, 12, 15, 18, 21, 24, 27 deaths +1 and assists +2
            k = list(map(int, [kda_no_slash[0], kda_no_slash[3], kda_no_slash[6], kda_no_slash[9], kda_no_slash[12], kda_no_slash[15], kda_no_slash[18], kda_no_slash[21], kda_no_slash[24], kda_no_slash[27]])) # typecasts kills to singular list based off index locations (assumes no extra characters are output)
            d = list(map(int, [kda_no_slash[1], kda_no_slash[4], kda_no_slash[7], kda_no_slash[10], kda_no_slash[13], kda_no_slash[16], kda_no_slash[19], kda_no_slash[22], kda_no_slash[25], kda_no_slash[28]])) 
            a = list(map(int, [kda_no_slash[2], kda_no_slash[5], kda_no_slash[8], kda_no_slash[11], kda_no_slash[14], kda_no_slash[17], kda_no_slash[20], kda_no_slash[23], kda_no_slash[26], kda_no_slash[29]]))
            stats_dict['K'] = k # adding kills to dict
            stats_dict['D'] = d # adding deaths to dict
            stats_dict['A'] = a # adding assists to dict
            stats_dict['KD'] = np.divide(k,d) # creating KD ratio by element wise division
            stats_dict['KDA'] = np.divide(np.add(k, a), d) # creating KDA by element wise multiplication and division
        else: # normal column of stats can append to dictionary and process with GCV
            stats_dict[stat] = google_cloud_vision(stat+'.png')[1:] # adds GCV output to dictionary at the current stat
    stats_dict['SID'] = [SID for i in range(10)] # populates stats_dict with list of the scrim id (scrim id is MONTHDDYYYYMAPDURATION) example: OCT302022BREEZE2356
    stats_dict['MAP'] = [MAP for i in range(10)] # populates stats_dict with list of the map to make for better groups joins and pivots

    try:
        SB = pd.DataFrame.from_dict(stats_dict)
    except:
        print(f'Shape is not correct, cannot cast into DataFrame. GCV likely output unexpected character. Check output and configure upscale factor. Scoreboard dict : {stats_dict}')
    
    return SB


if __name__ == "__main__":    
    read_screenshots('scorescreen.png', 'summary.png')
