import numpy as np
import faiss
import random
import os
import time
import h5py


if not os.path.exists("train_data.npy"):
    print("Generate Data...")
    vectors = []
    count = 0
     
    with open('file_name', 'r') as f:
        data = f.readlines()
        num_query = len(data)
            
        for line in data:
            query = list(line.split()[1:(len(line.split()))])
            vector = [float(str) for str in query]
            vectors.append(vector)
            count = count + 1
            #print(int(count))
            if (count % 1000 == 0):
                print("Progress: {0}/{1}".format(count, num_query), end="\r")

    random.shuffle(vectors)

    train_data = np.array(vectors[1000:]).astype('float32')
    test_data = np.array(vectors[0:1000]).astype('float32')

    print("Save Data...")
    np.save('train_data',train_data)
    np.save('test_data',test_data)

else:
    print("Load Data...")
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')

print(train_data.shape)
print(test_data.shape)

d = 1200
k = 1000 


index = faiss.IndexFlatL2(d)
index.add(train_data)
print(index.ntotal)
start = time.time()
D, I = index.search(test_data, k)   
end = time.time()
print(end-start)
#print(I[:5])                   
#print(D[:5])                   

f = h5py.File("mojie-1200-euclidean", 'w')
f.create_dataset("train", data=train_data)
f.create_dataset("test", data=test_data)
f.create_dataset("distances", data=D)
f.create_dataset("neighbors", data=I)
f.attrs["distance"] = "euclidean"
f.close()

hdf5_f = h5py.File("mojie-1200-euclidean")
distance = hdf5_f.attrs['distance']
