# U-Net for Glomeruli segmentation in H&E WSI
U-Net model for glomeruli segmentation in H&E stained Whole Slide Images.

# Usage
Segment glomeruli in a kidney H&E stained WSI using a pre-trained U-Net model.
This generates a glomeruli segmentation mask that is stored in zarr format.
The segmentation mask is stored inside a group called `class`.
```
python segment.py -m /path/to/checkpoint -i /path/to/zarr/files -o /ouput/directory
```
If this is run on a machine with GPUs, the size of the processed chunks can be modified to make the segmentation more efficient.
This is limited by the GPU's memory.
```
python segment.py -m /path/to/checkpoint -i /path/to/zarr/files -o /ouput/directory -cs 2048
```
By default, only the class (Glomeruli/Background) are stored in the output file.
The option `-sp` can be used to store the prediction probabilities.
These will be stored in a separate group called `probs`.
```
python segment.py -m /path/to/checkpoint -i /path/to/zarr/files -o /ouput/directory -sp
```
