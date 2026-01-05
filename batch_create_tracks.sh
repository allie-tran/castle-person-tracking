for day in day1 day2 day3 day4
do
    for people in Allie Bjorn Cathal Florian Klaus Luca Onanong Stevan Tien Werner
    # for people in Kitchen Living1 Living2 Meeting
    do
        if [ -e person_tracking_output/"$day"_"$people"/final_metadata.json ]; then
            echo "Skipping $day $people, already done"
            continue
        fi
        python3 track_and_describe.py --person "$people" --day "$day" --no_video --no_descriptions --output_folder person_tracking_output --input_folder /mnt/ssd0/Images/CASTLE/
    done
done
