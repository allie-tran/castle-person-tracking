#!/bin/bash
VIDEO_DIR=/mnt/ssd0/castle/castle_downloader/CASTLE2024/main/

for day in day1 day2 day3 day4
do
    for people in Allie Bjorn Cathal Florian Klaus Luca Onanong Stevan Tien Werner Kitchen Living1 Living2 Meeting
    do
        for hour in 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22
        do
            if [ -e video_output/"$day"_"$people"/"$hour"_final_metadata.json ]; then
                echo "Skipping $day $people $hour, already done"
                continue
            fi
            python3 person_track_video.py --person "$people" --day "$day" --hour "$hour" --no_video --output_folder video_output --video "$VIDEO_DIR"/"$day"/"$people"/"$hour".mp4
        done
    done
done
