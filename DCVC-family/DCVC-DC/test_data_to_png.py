# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from src.utils.video_reader import YUVReader
from src.utils.video_writer import PNGWriter


def convert_one_seq_to_png(src_path, width, height, dst_path):
    src_reader = YUVReader(src_path, width, height, src_format='420')
    png_writer = PNGWriter(dst_path, width, height)
    rgb = src_reader.read_one_frame(dst_format='rgb')
    processed_frame = 0
    while not src_reader.eof:
        png_writer.write_one_frame(rgb=rgb, src_format='rgb')
        processed_frame += 1
        rgb = src_reader.read_one_frame(dst_format='rgb')
    print(src_path, processed_frame)


def main():
    root_path = "/remote-home/shuwen/media/data/"
    dataset = "HEVC_B/"
    src_path = "BasketballDrive_1920x1080_50.yuv"
    width = 1920
    height = 1080
    dst_path = "BasketballDrive_1920x1080_50/"
    convert_one_seq_to_png(src_path, width, height, dst_path)


if __name__ == "__main__":
    main()
