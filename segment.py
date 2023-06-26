import os
import logging

import torch

import zarr
import dask
import dask.array as da
import numpy as np


from models import UNet


def predict_image(input_fn, output_dir, predict_chunk_func, threshold=0.5, 
                  chunk_size=1024,
                  save_probs=True):
    basename_split = input_fn.split(".zarr")
    component = basename_split[1]
    basename = os.path.basename(basename_split[0]) + ".zarr"

    z = da.from_zarr(input_fn)

    # Verify that the image is CYX
    if z.ndim > 3:
        z = z[0, :, 0, ...]

    z = da.rechunk(z, (3, chunk_size, chunk_size))

    z_pred = z.map_overlap(predict_chunk_func,
                           depth=(1, 16, 16),
                           dtype=np.float32,
                           drop_axis=(0,),
                           meta=np.empty((0,), dtype=np.float32))

    # Save the prediction probabilities when specified by the user
    if save_probs:
        z_pred.to_zarr(os.path.join(output_dir, basename + ".zarr"),
                       component="probs/" + component,
                       compressor=zarr.Blosc())

    z_class = z_pred > threshold
    z_class.to_zarr(os.path.join(output_dir, basename + ".zarr"),
                    component="class/" + component,
                    compressor=zarr.Blosc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("U-Net for glomeruli segmentation")
    parser.add_argument("-i", "--inputs", dest="inputs", type=str, nargs="+",
                        required=True,
                        help="Input images in zarr format in TCZYX axes "
                             "ordering (usual bioformats2raw axes ordering)")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                        default="./",
                        help="Output directory where to save the predictions")
    parser.add_argument("-m", "--model", dest="model", type=str, required=True,
                        help="Model checkpoint containing the U-Net weights")
    parser.add_argument("-t", "--threshold", dest="threshold", type=float,
                        default=0.5,
                        help="Threshold applied to the predictions made by the"
                             " model to consider them glomeruli or not")
    parser.add_argument("-cs", "--chunk-size", dest="chunk_size", type=int,
                        default=1024,
                        help="Size of the chunks processed by the model")
    parser.add_argument("-sp", "--save-probs", dest="save_probs",
                        action="store_true",
                        default=False,
                        help="Enable saving the prediction probabilities")

    args = parser.parse_args()

    logger = logging.getLogger('segmentation_log')
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    logger.addHandler(console)

    unet_model = UNet()

    map_to_device = "cpu" if not torch.cuda.is_available() else None
    checkpoint = torch.load(args.model, map_location=map_to_device)

    unet_model.load_state_dict(checkpoint)
    unet_model.eval()
    logger.info(f"Loaded U-Net model from {args.model}")

    unet_model = torch.nn.DataParallel(unet_model)
    if torch.cuda.is_available():
        unet_model.cuda()

    def predict_chunk(chunk):
        x = torch.from_numpy(chunk)
        x = x[None, ...].float() / 255.0

        with torch.no_grad():
            pred = unet_model(x)
            pred = pred[0, 0].cpu().sigmoid().numpy()

        return pred

    for in_fn in args.inputs:
        output_fn = predict_image(in_fn, args.output_dir, predict_chunk,
                                  chunk_size=args.chunk_size,
                                  save_probs=args.save_probs)
        logger.info(f"Segmented {in_fn}, output was saved to {output_fn}")

    logging.shutdown()
