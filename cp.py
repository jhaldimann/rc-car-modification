import cv2 as cv
import numpy as np
import time
import ray

def to_binary(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print(thresh)
    return binary

def erode(image, size):
    dst = cv.erode(image, np.ones((size,size), 'uint8'), iterations=5)
    return dst

def dilate(image, size):
    dst = cv.dilate(image, np.ones((size,size), 'uint8'), iterations=1)
    return dst

def fc_process(base, dst, start, end, margin):
    s = time.time()
    start_old = 0
    c = 0
    for y, row in enumerate(base[start:end]):
        x_start = None
        x_end = None
        if y % 10 != 0:
            continue
        for x, pixel in enumerate(row):
            if x < start_old - margin:
                continue
            if pixel == 0 and x_start is None:
                x_start = x
                start_old = x_start
            elif x_start is not None and pixel == 255:
                x_end = x - 1
                center = int(x_start + (x_end - x_start) / 2)
                y_value = y + start
                dst[y_value][center] = 0
                dst[y_value][center-1] = 0
                dst[y_value][center+1] = 0
                dst[y_value][center-2] = 0
                dst[y_value][center+2] = 0
                dst[y_value][center-3] = 0
                dst[y_value][center+3] = 0
                break
            c += 1
    e = time.time()
    print(f"Elapsed time {start}-{end}: {(e - s) * 1000} ms")
    print(f"Iterations {start}-{end}: {c}")


def find_center_fast(image):
    s = time.time()
    dst = np.full((len(image), len(image[0])), 255, 'uint8')
    margin = int(len(image[0]) / 100)

    part = len(image) / 4


    fc_process(image, dst, 0, len(image), margin)

    e = time.time()
    print(f"Elapsed time: {(e - s) * 1000} ms")
    return dst, e - s

def find_center_mp(image):
    s = time.time()
    dst = np.full((len(image), len(image[0])), 255, 'uint8')
    margin = int(len(image[0]) / 100)

    cpu_count = mp.cpu_count()

    part = int(len(image) / cpu_count)
    print(cpu_count)

    queue = mp.Queue()

    processes = []

    for i in range(cpu_count-1):
        processes.append(mp.Process(target=fc_process, args=(image, dst, part*i+1, part*(i+1), margin)))
    for process in processes:
        print(time.time() - s)
        process.start()
    fc_process(image, dst, part * (cpu_count-1) + 1, len(image), margin)
    for process in processes:
        process.join()

    e = time.time()
    print(f"Elapsed time: {(e - s) * 1000} ms")
    return dst, e - s

def find_center_slow(image):
    s = time.time()
    dst = np.full((len(image), len(image[0])), 255, 'uint8')
    start_old = 0
    margin = int(len(image[0]) / 100)
    for y, row in enumerate(image):
        start = None
        end = None
        for x, pixel in enumerate(row):
            if pixel == 0 and start is None:
                start = x
            elif start is not None and pixel == 255:
                end = x - 1
                center = int(start + (end - start) / 2)
                dst[y][center] = 0
                dst[y][center-1] = 0
                dst[y][center+1] = 0
                dst[y][center-2] = 0
                dst[y][center+2] = 0
                dst[y][center-3] = 0
                dst[y][center+3] = 0
                break
    e = time.time()
    print(f"Elapsed time: {(e - s) * 1000} ms")
    return dst

"""
Test
"""
if __name__ == '__main__':
    left = cv.imread('test-input/left-curve.png')
    right = cv.imread('test-input/right-curve.png')
    straight = cv.imread('test-input/straight.png')

    b_left = erode(to_binary(left), 5)
    b_right = erode(to_binary(right), 5)
    b_straight = erode(to_binary(straight), 5)

    cv.imwrite('test-output/e-left-curve.png', b_left)
    cv.imwrite('test-output/e-right-curve.png', b_right)
    cv.imwrite('test-output/e-straight.png', b_straight)

    f_times = []
    mt_times = []
    #for i in range(100):
        #_, t = find_center_fast(b_left)
        #f_times.append(t)
    _, t = find_center_mp(b_left)
    mt_times.append(t)

    #print(f"Average time: Fast = {sum(f_times)/len(f_times)*1000} ms, MP = {sum(mt_times)/len(mt_times)*1000} ms")


    #cv.imwrite('test-output/cs-left-curve.png', find_center_slow(b_left))
    #cv.imwrite('test-output/cf-left-curve.png', find_center_fast(b_left))
    #cv.imwrite('test-output/cmp-left-curve.png', find_center_mp(b_left))
