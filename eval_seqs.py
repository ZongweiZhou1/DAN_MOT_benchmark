from tracker import SSTTracker, TrackerConfig, Track
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os
from utils.evaluation import Evaluator
import motmetrics as mm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(description='Evaluation of DAN on MOT')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--data_root', default='/data/zwzhou/Data', help='GT_ROOT')
parser.add_argument('--show_image', default=False, help='show image if true, or hidden')
parser.add_argument('--save_video', default=False, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')
parser.add_argument('--exp_name', default='02', help='the name of this experiment')
args = parser.parse_args()

def write_results(filename, results, data_type='mot'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for fid, tid, x1, y1, w, h in results:
            line = save_format.format(frame=fid, id=tid, x1=x1, y1=y1, w=w, h=h)
            f.write(line)


def track_seq(seq, det_type, choice_str):
    seq_name = 'MOT{}-{}{}'.format(args.mot_version, seq.zfill(2), det_type)
    dataset_image_folder_format = os.path.join(args.mot_root, 'train/{}/img1')

    detection_file_name_format = os.path.join(args.mot_root, 'train/{}/det/det.txt')

    save_folder = os.path.join(args.log_folder, choice_str)
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    saved_file_name_format = os.path.join(save_folder, '{}.txt')
    save_video_name_format = os.path.join(save_folder, '{}.avi')

    timer = Timer()

    image_folder = dataset_image_folder_format.format(seq_name)
    detection_file_name = detection_file_name_format.format(seq_name)
    saved_file_name = saved_file_name_format.format(seq_name)
    save_video_name = save_video_name_format.format(seq_name)

    print('start processing ' + seq_name)

    # tracker = SSTTracker()
    # reader = MOTDataReader(image_folder=image_folder, detection_file_name=detection_file_name, min_confidence=0.0)
    # result = list()
    # result_str = saved_file_name
    # first_run = True
    # for i, item in enumerate(reader):
    #     if i % 20 == 0:
    #         print('Processing frame {} ({:.2f} fps)'.format(i, 1./max(1e-5, timer.average_time)))
    #
    #     if i > len(reader):
    #         break
    #
    #     if item is None:
    #         continue
    #
    #     img = item[0]
    #     det = item[1]
    #
    #     if img is None or det is None or len(det) == 0:
    #         continue
    #
    #     if len(det) > config['max_object']:
    #         det = det[:config['max_object'], :]
    #
    #     h, w, _ = img.shape
    #
    #     if first_run and args.save_video:
    #         vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))
    #         first_run = False
    #
    #     det[:, [2, 4]] /= float(w)
    #     det[:, [3, 5]] /= float(h)
    #     timer.tic()
    #     image_org = tracker.update(img, det[:, 2:6], args.show_image, i)
    #     timer.toc()
    #
    #
    #     if args.show_image and not image_org is None:
    #         cv2.imshow('res', image_org)
    #         cv2.waitKey(1)
    #
    #     if args.save_video and not image_org is None:
    #         vw.write(image_org)
    #
    #     # save result
    #     for t in tracker.tracks:
    #         n = t.nodes[-1]
    #         if t.age == 1:
    #             b = n.get_box(tracker.frame_index - 1, tracker.recorder)
    #             result.append(
    #                 [i+1] + [t.id] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    #             )
    # # save data
    # write_results(saved_file_name, result)
    # print('Average time to process {}-{} is ({:.2f} fps)'.format(seq_name, det_type, 1. / max(1e-5, timer.average_time)))
    # print('The tracking results saved to {}'.format(result_str))
    return saved_file_name


def main(seqs=('2',), det_types=('',)):
    # run tracking
    accs = []
    data_root = args.data_root+'/MOT{}/train'.format(args.mot_version)
    choice = (0, 0, 4, 0, 3, 3)
    TrackerConfig.set_configure(choice)
    choice_str = TrackerConfig.get_configure_str(choice)
    seq_names = []
    for seq in seqs:
        for det_type in det_types:
            result_filename = track_seq(seq, det_type, choice_str)
            seq_name ='MOT{}-{}{}'.format(args.mot_version, seq.zfill(2), det_type)
            seq_names.append(seq_name)
            print('Evaluate seq:{}'.format(seq_name))

            evaluator = Evaluator(data_root, seq_name, 'mot')
            accs.append(evaluator.eval_file(result_filename))

    # get summary
    # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    metrics = mm.metrics.motchallenge_metrics
    # metrics = None
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seq_names, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(os.path.join(args.log_folder, choice_str),
                                                 'summary_{}.xlsx'.format(args.exp_name)))

if __name__=='__main__':
    seqs_str = '''2,5,9,11,13'''
    if args.mot_version==16:
        det_type=('',)
    elif args.mot_version==17:
        det_type = ('-FRCNN',)
    else:
        raise ('Wrong seqs')

    seqs = [seq.strip() for seq in seqs_str.split(',')]

    main(seqs=seqs, det_types=det_type)


