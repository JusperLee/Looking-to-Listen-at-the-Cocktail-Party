with open('./AV_model_database/dataset_train.txt', 'r') as tr:
    lines = tr.readlines()
    for line in lines:
        info = line.strip().split('.')
        num1 = info[0].strip().split('-')[1]
        num2 = info[0].strip().split('-')[2]
        new_line = line.strip() + ' ' + num1 + '_face_emb.npy' + ' ' + num2 + '_face_emb.npy\n'
        with open('AVdataset_train.txt', 'a') as f:
            f.write(new_line)

with open('./AV_model_database/dataset_val.txt', 'r') as val:
    lines = val.readlines()
    for line in lines:
        info = line.strip().split('.')
        num1 = info[0].strip().split('-')[1]
        num2 = info[0].strip().split('-')[2]
        new_line = line.strip() + ' ' + num1 + '_face_emb.npy' + ' ' + num2 + '_face_emb.npy\n'
        with open('AVdataset_val.txt','a') as f:
            f.write(new_line)