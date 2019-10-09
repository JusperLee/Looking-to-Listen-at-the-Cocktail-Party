import os, glob


def check_face_valid(index, part, check_pth):
    path = check_pth + '/frame_%d_%02d.jpg' % (index, part)
    if (not os.path.exists(path)):
        return False
    else:
        return True


if __name__ == "__main__":
    check_pth = './face_input'
    check_range = (0, 20)
    valid_face_txt = 'valid_face_text.txt'
    for i in range(check_range[0], check_range[1]):
        valid = True
        print('Processing Frame %s' % i)
        for j in range(1, 76):
            if (check_face_valid(i, j, check_pth) == False):
                path = check_pth + '/frame_%d_*.jpg' % i
                for file in glob.glob(path):
                    os.remove(file)
                valid = False
                print('Frame %s is not valid' % i)
                break

        if valid == True:
            with open(valid_face_txt,'a') as f:
                frame_name = 'frame_%d'%i
                f.write(frame_name+'\n')
