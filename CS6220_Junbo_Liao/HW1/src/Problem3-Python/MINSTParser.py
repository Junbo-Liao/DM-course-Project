import numpy
import struct

from sklearn.metrics import euclidean_distances


def loadImageSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = numpy.reshape(imgs, [imgNum, width * height])
    return imgs, head

file1= 'train-images.idx3-ubyte'
imgs,data_head = loadImageSet(file1)
n_row = imgs.shape[0]
f = open("MINST_dotProduct.txt",'wb')
for i in range(0,n_row):
    d = numpy.dot(imgs[i], imgs.T)
    numpy.savetxt(f,d,fmt='%.4f')
    print(i)
f.close
f = open("MINST_euclidiandist.txt",'wb')
for i in range(0,n_row):
    d = euclidean_distances(imgs[i], imgs)
    numpy.savetxt(f,d,fmt='%.4f')
    print(i)
f.close()