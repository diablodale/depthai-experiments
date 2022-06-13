#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')
parser.add_argument('-d', '--draw', default=True, action='store_true',
                    help='Draw on frames the sequence number and timestamp')
parser.add_argument('-v', '--verbose', default=0, action='count',
                    help='Verbose, -vv for more verbosity')
parser.add_argument('-t', '--dev_timestamp', default=False, action='store_true',
                    help='Get device timestamps, not synced to host. For debug')

args = parser.parse_args()

cam_list = ['right']
cam_socket_opts = {
    'rgb'   : dai.CameraBoardSocket.RGB,
    'left'  : dai.CameraBoardSocket.LEFT,
    'right' : dai.CameraBoardSocket.RIGHT,
    'depth' : dai.CameraBoardSocket.RIGHT,
}

def make_mono(cam_name, make_xout):
    global pipeline
    global cam
    global xout
    global cam_socket_opts
    if not cam_name in cam:
        cam[cam_name] = pipeline.create(dai.node.MonoCamera)
        cam[cam_name].setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
        cam[cam_name].setFps(args.fps)
    if make_xout and not (cam_name in xout):
        xout[cam_name] = pipeline.create(dai.node.XLinkOut)
        xout[cam_name].setStreamName(cam_name)
        #xout[cam_name].input.setQueueSize(1)
        #xout[cam_name].input.setBlocking(False)
    if make_xout:
        cam[cam_name].out.link(xout[cam_name].input)

# Start defining a pipeline
pipeline = dai.Pipeline()

cam = {}
xout = {}
for c in cam_list:
    xout[c] = pipeline.create(dai.node.XLinkOut)
    xout[c].setStreamName(c)
    #xout[c].input.setQueueSize(1)
    #xout[c].input.setBlocking(False)
    if c == 'rgb':
        cam[c] = pipeline.create(dai.node.ColorCamera)
        cam[c].setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam[c].setIspScale(4, 9)  # downsample 853.3x480
        cam[c].isp.link(xout[c].input)
        cam[c].setBoardSocket(cam_socket_opts[c])
        cam[c].setFps(args.fps)
    elif c == 'depth':
        make_mono('left', False)
        make_mono('right', False)
        cam[c] = pipeline.create(dai.node.StereoDepth)
        # settings
        cam[c].useHomographyRectification(False)
        cam[c].setSubpixel(True)
        cam[c].setDepthAlign(dai.CameraBoardSocket.RGB)
        cam[c].setLeftRightCheck(True)
        cam[c].setOutputSize(832, 480)
        cam[c].setOutputKeepAspectRatio(True)
        cam[c].setExtendedDisparity(False)
        cam[c].setRectifyEdgeFillColor(0)
        # links
        cam['left'].out.link(cam[c].left)
        cam['right'].out.link(cam[c].right)
        #cam[c].left.setQueueSize(1);  # https://discord.com/channels/790680891252932659/924798503270625290/961835869352886292
        #cam[c].right.setQueueSize(1)
        #cam[c].left.setBlocking(False)
        #cam[c].right.setBlocking(False)
        cam[c].depth.link(xout[c].input)
    else:
        make_mono(c, True)

def get_seq(packet):
    return packet.getSequenceNum()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    allowed_instances = cam_list

    def __init__(self):
        self.seq_packets = {}
        self.last_paired_seq = None

    def add_packet(self, packet, cam_name):
        if packet is not None and cam_name in self.allowed_instances:
            seq_key = get_seq(packet)
            self.seq_packets[seq_key] = {
                **self.seq_packets.get(seq_key, {}),
                cam_name: packet
            }
            #print("added " + cam_name)
        elif packet is None:
            pass #print(" empty", end = '')
        else:
            pass #print("cant add " + cam_name + " unexpected")

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.allowed_instances):
                results.append(self.seq_packets[key])
                self.last_paired_seq = key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    tstart = time.monotonic()  # applied as an offset

    saveidx = 0

    q = {}
    for c in cam_list:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False) # shrink!!
    ps = PairingSystem()

    window = ' + '.join(cam_list)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, (len(cam_list)*640, 480))

    if args.verbose > 0:
        print("   Seq  Left_tstamp  RGB-Left  Right-Left  Dropped")
        print("   num    [seconds]  diff[ms]    diff[ms]  delta")

    seq_prev = -1
    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        for c in cam_list:
            ps.add_packet(q[c].tryGet(), c)

        for synced in ps.get_pairs():
            frame, seqnum, tstamp = {}, {}, {}
            for c in cam_list:
                pkt = synced[c]
                frame[c] = pkt.getCvFrame()
                if c == 'depth':
                    d_min = np.min(frame[c])
                    d_max = 5000 #np.max(frame[c])
                    frame[c] = (frame[c] - d_min) / (d_max - d_min)
                    frame[c] = np.array(frame[c]) * 255
                    frame[c] = frame[c].astype(np.uint8)
                    frame[c] = cv2.applyColorMap(frame[c], cv2.COLORMAP_BONE)
                elif c == 'rgb':
                    pass
                else:
                    frame[c] = cv2.cvtColor(frame[c], cv2.COLOR_GRAY2BGR)
                seqnum[c] = pkt.getSequenceNum()
                if args.dev_timestamp:
                    tstamp[c] = pkt.getTimestampDevice().total_seconds()
                else:
                    tstamp[c] = pkt.getTimestamp().total_seconds() - tstart
                if args.draw:
                    text = '{:1d}'.format(seqnum[c]) + '  {:.6f}'.format(tstamp[c])
                    cv2.putText(frame[c], text, (8,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 8, cv2.LINE_AA)
                    cv2.putText(frame[c], text, (8,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            first_seqnum = list(seqnum.values())[0]
            if not(all(value == first_seqnum for value in seqnum.values())):
                print('ERROR: out of sync!!!')
            if args.verbose > 0:
                seq = seqnum['left']
                seq_diff = seq - seq_prev
                seq_prev = seq
                rgb_left_diff   = (tstamp['rgb']   - tstamp['left']) * 1000
                right_left_diff = (tstamp['right'] - tstamp['left']) * 1000
                if (args.verbose > 1 or seq_diff != 1
                                     or abs(rgb_left_diff)   > 0.15
                                     or abs(right_left_diff) > 0.05):
                    print('{:6d} {:12.6f} {:9.3f}   {:9.3f}'.format(
                          seq, tstamp['left'], rgb_left_diff, right_left_diff), end='')
                    if seq_diff != 1: print('   ', seq_diff - 1, end='')
                    print()
            frame_final = np.hstack(([frame[c] for c in cam_list]))
            cv2.imshow(window, frame_final)

            key = cv2.waitKey(1)
            if key == ord('c'):
                filename = f'capture_{saveidx:04}.png'
                cv2.imwrite(filename, frame_final)
                print('Saved to:', filename)
                saveidx += 1
            elif key == ord('q'):
                quit()
