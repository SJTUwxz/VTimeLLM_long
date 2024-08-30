# NextQA evaluation phase

## save predicted frames to folder, should use eval.py but since inference are all done, use pred2frames.py instead to process the saved predictions

### current predicted frames saved directory '$VL_DATA_DIR/predicted_best_frames'
nextqa_val:
6-frames question + answer: next_qa_6-frame_answer/vicuna-v1-5-7b_8_1e-4-stage2-answer
6-frames question only: next_qa_6-frame_question/vicuna-v1-5-7b_8_1e-4-stage2-question
6-frames simple instruction: next_qa/vicuna-v1-5-7b_8_1e-4-stage2

nextqa_traintest:
6-frames question + answer: next_qa_traintest_6-frame_answer/vicuna-v1-5-7b_8_1e-4-stage2-answer
6-frames question only: next_qa_traintest_6-frame_question/vicuna-v1-5-7b_8_1e-4-stage2-question
6-frames simple instruction: next_qa_traintest/vicuna-v1-5-7b_8_1e-4-stage2

### 1/2/3 frames and segment predictions saved directories that needs to be transformed into 6-frame predictions '$VL_DATA_DIR/vtimellm_predictions', predictions are in percentage format
nextqa_val:
next_qa_{frame_number}-frame_[question, answer]
next_qa_traintest_{frame_number}-frame_[question, answer]

next_qa_segment_[question, answer]/
next_qa_traintest_segment_[question, answer]/vicuna-v1-5-7b_8_1e-4-answer-segment/


```bash

python unify_scripts/pred2frames.py --frames_num 1 --question_or_answer question --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 1 --question_or_answer question --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 1 --question_or_answer answer --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 1 --question_or_answer answer --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 2 --question_or_answer question --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 2 --question_or_answer question --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 2 --question_or_answer answer --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 2 --question_or_answer answer --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 3 --question_or_answer question --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 3 --question_or_answer question --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 3 --question_or_answer answer --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --frames_num 3 --question_or_answer answer --subset traintest --missing_prediction_complete uniform

# segment
python unify_scripts/pred2frames.py --predict_segment --question_or_answer question --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --predict_segment --question_or_answer question --subset traintest --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --predict_segment --question_or_answer answer --subset val --missing_prediction_complete uniform

python unify_scripts/pred2frames.py --predict_segment --question_or_answer answer --subset traintest --missing_prediction_complete uniform


## move on to ig-vlm vtimellm_scripts.md for evaluation

```
