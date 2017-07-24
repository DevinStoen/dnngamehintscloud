from __future__ import print_function
import sys
from flask import Flask
from flask import request
import os
from use_archive import unzip_archive
from example import classify
import datetime

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
    #username = request.args.get('username')###
    #print(username)

    imgData = request.get_data()
    convertImage(imgData)

    #folderPath = "/home/user/Desktop/CommunicatC#/"+username 
    #textPath = "/home/user/Desktop/CommunicatC#/"+username +"/"+username + "_result.txt"
    folderPath = ""
    textPath = ""
    #if not os.path.isdir(folderPath):##do not have folder for user
     #   os.makedirs(folderPath) 
    #gamename = request.args.get('gamename')

    #if 'file' not in request.files:
    #    flash('No file part')
    #    return redirect(request.url)
    #open target file to record results
    #file = request.files['file']
    #target = open(textPath,"a")
    #imageName = str(datetime.datetime.now())
    #print (imageName+"\n")
    #target.write("----receive the image file time-----" + imageName + "\n")
    #print(file.filename, file=sys.stderr)
    #file.save(os.path.join(folderPath,imageName))
    #target.write("----saved image file time----------" + str(datetime.datetime.now())+ "\n")
    ##Generate image file    
    #image_file = [folderPath+"//"+imageName]

    image_file = cv2.imread('output.png')
    print('read image')
    ##change gamename
    #if gamename == "INSIDE":
    archive = '/INSIDE_v5.1_AlexNet.tar.gz'
    #elif gamename == "Portal": 
        #archive = '/home/user/Desktop/CommunicatC#/Portal_v5.tar.gz'
    ##write log result
    #target.write("----Start get classified result------" + str(datetime.datetime.now()) + "\n")
    resultLabel = classify_archive(archive,image_file,folderPath,textPath,target)
    #target.write("----Receive classified result--------" + str(datetime.datetime.now()) + "\n")
    #target.write("___________________________________________________________________\n")
    #target.close()
    print(resultLabel, file=sys.stderr)
    #print ("----Send resultLabel---" + str(datetime.datetime.now())+ "\n")
    return resultLabel

## get classify result
def classify_archive(archive,image_file,folderPath,textPath,target):
    #archive = '/home/user/Desktop/CommunicatC#/Portal_v5.tar.gz'
    #image_file = ['/home/user/Desktop/CommunicatC#//file.png'] 
    batch_size= None
    use_gpu= True
    tmpdir = unzip_archive(archive)

    caffemodel = None
    deploy_file = None
    mean_file = None
    labels_file = None
    for filename in os.listdir(tmpdir):
        full_path = os.path.join(tmpdir, filename)
        if filename.endswith('.caffemodel'):
            caffemodel = full_path
        elif filename == 'deploy.prototxt':
            deploy_file = full_path
        elif filename.endswith('.binaryproto'):
            mean_file = full_path
        elif filename == 'labels.txt':
            labels_file = full_path
        #else:
            #print ('Unknown file:', filename, file=sys.stderr)#################33
    assert caffemodel is not None, 'Caffe model file not found'
    assert deploy_file is not None, 'Deploy file not found'
    #print("NOt working: print Image file before call classify.\n")
    #print(image_file)
    resultLabel = classify(target, folderPath, textPath, caffemodel, deploy_file, image_file,
         mean_file=mean_file, labels_file=labels_file,
        batch_size=batch_size, use_gpu=use_gpu)
    return resultLabel


def convertImage(imgData1):
    #imgstr = re.search(r'base64,(.*)',imgData1).group(1)
    #imgstr = int(imgData1, base=2)
    #print(imgstr)
    #imgstr = numpy.unpackbits(Bytes)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgData1)) 

if __name__ == "__main__":
    app.run(host = "0.0.0.0")
