#!/bin/bash

checkpoint=$1
datadir=$2
resultdir=$3

MAXDUR=9.6

# Check if the epoch exists
if [ ! -f ${checkpoint} ]; then
  echo "ERROR: ${checkpoint} does not exist"
  exit 1;
fi

# Cut long audio to max length
normdir=${datadir%/}_lt${MAXDUR}
if [ ! -d $normdir ]; then
  mkdir -p $normdir
  while read line; do
    dur=$( soxi -D $line )
    echo $line $dur
    if (( $(echo "$dur > $MAXDUR" | bc -l) )); then
      bname=$( basename $line .wav )
      sox $line $normdir/${bname}_PART.wav trim 0 $MAXDUR : newfile : restart
    else
      cp $line $normdir
    fi
  done < <( find ${datadir}/ -name "*.wav" )
fi

# Prepare script file
mkdir -p $resultdir/sort
find $normdir/ -name "*.wav" > $resultdir/wav_lt${MAXDUR}.scp

python inference_FDN.py --resume_ckpt $checkpoint --save_path $resultdir --scp $resultdir/wav_lt${MAXDUR}.scp

sort -t' ' -k1,1 -k2n,2 $resultdir/unit0.02.score > $resultdir/sort/unit0.02.score
python normalize_score.py $resultdir/sort/unit0.02.score $resultdir/unit0.02.score

