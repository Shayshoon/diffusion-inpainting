#!/bin/bash

echo running mask blur
python ./PipelineTester.py --pipeline MaskBlur --skip-existing --src samples/ --dst pipe_results/
echo running copy and blur
python ./PipelineTester.py --pipeline CopyAndBlur --skip-existing --src samples/ --dst pipe_results/

echo evaluating
python ./PipelineTester.py --evaluate

mv pipe_results/BackgroundReconstruction/ pipe_results/BackgroundReconstructionOld

echo running copy and blur
python ./PipelineTester.py --pipeline BackgroundReconstruction --src samples/ --dst pipe_results/

