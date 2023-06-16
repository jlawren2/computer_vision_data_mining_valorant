# GoogleCloudVision_On_Valorant_Scrims

## FOLLOW *some* OF THE INSTRUCTIONS ON THIS PAGE FIRST: https://cloud.google.com/vision/docs/ocr
If you plan on running this you will need to follow the instructions on that page.

1. Setup your enviroment for Google Cloud Vision. There are a few packages to install and some imports I use you will need to install.
2. (numpy, pandas, skimage: data, io) are the imports I used. May require pip install.
3. Figure out how to store your API key. Google has a link on best practice on where to store the .json file and/or API key. Google advises against storing it on your local computer but it can be done.
4. After configuring your enviroment and locating the API key as an enviromental variable you are ready to parse scrim data.
5. I used print screen on two screens scoreboard and summary of the post match screen. I pasted them in paint and saved them as scoreboard.png and summary.png respectively.
6. Run the python program. It will be output into two csv's. One is more of a team stats intended csv and the other individual stats.
7. Read this into a pandas dataframe and combine it with other exports.
