for day in day2 day1 day3 day4
do
    for people in Allie Bjorn Cathal Florian Klaus Luca Onanong Stevan Tien Werner
    # for people in Kitchen Living1 Living2 Meeting
    do
        python3 track_and_describe.py --person "$people" --day "$day"
    done
done
