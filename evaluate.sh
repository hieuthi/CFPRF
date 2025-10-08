#!/bin/bash

scorefile=$1
labelfile=$2
resultdir=$3


mkdir -p ${resultdir}/{utt_unit0.02,unit0.02}

echo "Calculate 20ms frame-based EER"
python partialspoof-metrics/calculate_eer.py --labpath ${labelfile} \
                                    --scopath ${scorefile} \
                                    --savepath ${resultdir}/unit0.02 \
                                    --unit 0.02 \
                                    --scoreindex 3 \
				    --negative_class \


echo "Calculate upscale utterance EER from 20ms-score"
python partialspoof-metrics/calculate_eer.py --labpath ${labelfile} \
                                    --scopath ${scorefile} \
                                    --savepath ${resultdir}/utt_unit0.02 \
                                    --unit 0.02 \
                                    --scoreindex 3 \
				    --negative_class \
				    --zoom 0

