# question-only: True if only question is used to find the best frames. 
# predict_frames_number: how many frames are Predicted
# train: True if generate annotation for training set

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 1 --train False

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 2 --train False

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 3 --train False

python nextqa_scripts/annotation_frames.py --question_only False --predict_frames_number 1 --train False

python nextqa_scripts/annotation_frames.py --question_only False --predict_frames_number 2 --train False

python nextqa_scripts/annotation_frames.py --question_only False --predict_frames_number 3 --train False

python nextqa_scripts/annotation_segment.py --question_only True --train False

python nextqa_scripts/annotation_segment.py --question_only False --train False

python nextqa_scripts/annotation_frames.py --question_only False --predict_frames_number 6 --train False

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 6 --train False

# generate annotations on train subset , subset needs to set traintest = True

python nextqa_scripts/annotation_no_instruction.py 

python nextqa_scripts/annotation_frames.py --question_only False --predict_frames_number 6 --train True --traintest True

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 6 --train True --traintest True

# generate annotations on train set
#
python nextqa_scripts/annotation_segment.py --question_only True --train True

python nextqa_scripts/annotation_segment.py --question_only False --train True

python nextqa_scripts/annotation_frames.py --question_only True --predict_frames_number 6 --train True --sample_size 20
