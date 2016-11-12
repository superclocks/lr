writer = file('mnist_train_libsvm.txt', 'w')
for line in file('mnist_train.txt'):
    ele = line.split(' ')
    writer.write(ele[0])
    s = ''
    for i in xrange(1, len(ele)):
        if float(ele[i]) != 0.0:
            s =  s + ' ' + str(i) + ':' + ele[i]
    writer.write(s + '\n')
writer.close()