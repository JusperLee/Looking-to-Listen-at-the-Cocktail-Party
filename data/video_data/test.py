from mtcnn.mtcnn import MTCNN
import cv2
import pandas as pd
import os

cat_train = pd.read_csv('../csv/avspeech_train.csv', header=None)
frame_path = './frames/'
output_dir = './face_input'
detect_range = (0,5)

if not os.path.isdir('./face_input'):
    os.mkdir('./face_input')

def bounding_box_check(faces,x,y):
    # check the center
    for face in faces:
        bounding_box = face['box']
        if(bounding_box[1]<0):
            bounding_box[1] = 0
        if(bounding_box[0]<0):
            bounding_box[0] = 0
        if(bounding_box[0]-50>x or bounding_box[0]+bounding_box[2]+50<x):
            print('change person from')
            print(bounding_box)
            print('to')
            continue
        if (bounding_box[1]-50 > y or bounding_box[1] + bounding_box[3]+50 < y):
            print('change person from')
            print(bounding_box)
            print('to')
            continue
        return bounding_box

def face_detect(file,detector,frame_path=frame_path,cat_train=cat_train):
    name = file.replace('.jpg', '').split('-')
    log = cat_train.iloc[int(name[0])]
    x = log[3]
    y = log[4]

    img = cv2.imread('%s%s'%(frame_path,file))
    x = img.shape[1] * x
    y = img.shape[0] * y
    faces = detector.detect_faces(img)
    # check if detected faces
    if(len(faces)==0):
        print('no face detect: '+file)
        return #no face
    bounding_box = bounding_box_check(faces,x,y)
    if(bounding_box == None):
        print('face is not related to given coord: '+file)
        return
    print(file," ",bounding_box)
    print(file," ",x, y)
    crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
    crop_img = cv2.resize(crop_img,(160,160))
    cv2.imwrite('%s/frame_'%output_dir + name[0] + '_' + name[1] + '.jpg', crop_img)
    #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    #plt.imshow(crop_img)
    #plt.show()


detector = MTCNN()
for i in range(detect_range[0],detect_range[1]):
    for j in range(1,76):
        file_name = "%d-%02d.jpg"%(i, j)
        if (not os.path.exists('%s%s' % (frame_path, file_name))):
            print('cannot find input: ' + '%s%s' % (frame_path, file_name))
            continue
        face_detect(file_name, detector)



