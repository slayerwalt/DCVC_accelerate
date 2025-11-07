# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.layers.cuda_inference import replicate_pad


class DMCIWrapper(nn.Module):
    """Wrapper for DMCI model to enable flops profiling"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, qp=32):
        """Forward pass for I-frame model"""
        curr_q_enc = self.model.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.model.q_scale_dec[qp:qp+1, :, :, :]
        
        # Encoder
        y = self.model.enc(x, curr_q_enc)
        y_pad = self.model.pad_for_y(y)
        z = self.model.hyper_enc(y_pad)
        z_hat = torch.round(z)
        
        # Hyper decoder and prior fusion
        params = self.model.hyper_dec(z_hat)
        params = self.model.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        
        # Prior processing (simplified, without entropy coding)
        q_enc, q_dec, scales, means = self.model.separate_prior(params, False)
        common_params = self.model.y_spatial_prior_reduction(params)
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.model.get_mask_4x(B, C, H, W, y.dtype, y.device)
        
        y = y * q_enc
        y_hat_0 = self.model.process_with_mask(y, scales, means, mask_0)[2]
        y_hat_so_far = y_hat_0
        
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.model.y_spatial_prior(self.model.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1 = self.model.process_with_mask(y, scales, means, mask_1)[2]
        y_hat_so_far = y_hat_so_far + y_hat_1
        
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.model.y_spatial_prior(self.model.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2 = self.model.process_with_mask(y, scales, means, mask_2)[2]
        y_hat_so_far = y_hat_so_far + y_hat_2
        
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.model.y_spatial_prior(self.model.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3 = self.model.process_with_mask(y, scales, means, mask_3)[2]
        y_hat = (y_hat_so_far + y_hat_3) * q_dec
        
        # Decoder
        x_hat = self.model.dec(y_hat, curr_q_dec)
        return x_hat


class DMCWrapper(nn.Module):
    """Wrapper for DMC model to enable flops profiling"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, qp=32):
        """Forward pass for P-frame model"""
        q_encoder = self.model.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.model.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.model.q_feature[qp:qp+1, :, :, :]
        
        # Feature extraction (using apply_feature_adaptor to get feature from reference frame)
        feature = self.model.apply_feature_adaptor()
        ctx, ctx_t = self.model.feature_extractor(feature, q_feature)
        
        # Encoder
        y = self.model.encoder(x, ctx, q_encoder)
        hyper_inp = self.model.pad_for_y(y)
        z = self.model.hyper_encoder(hyper_inp)
        z_hat = torch.round(z)
        
        # Prior parameter decoder
        hierarchical_params = self.model.hyper_decoder(z_hat)
        temporal_params = self.model.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.model.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        
        # Prior processing (simplified, without entropy coding)
        y, q_dec, scales, means = self.model.separate_prior_for_video_encoding(params, y)
        B, C, H, W = y.size()
        mask_0, mask_1 = self.model.get_mask_2x(B, C, H, W, y.dtype, y.device)
        
        y_hat_0 = self.model.process_with_mask(y, scales, means, mask_0)[2]
        cat_params = torch.cat((y_hat_0, params), dim=1)
        scales, means = self.model.y_spatial_prior(cat_params).chunk(2, 1)
        y_hat_1 = self.model.process_with_mask(y, scales, means, mask_1)[2]
        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec
        
        # Decoder and reconstruction
        feature = self.model.decoder(y_hat, ctx, q_decoder)
        x_hat = self.model.recon_generation_net(feature, self.model.q_recon[qp:qp+1, :, :, :])
        return x_hat


def profile_dmci_model(batch_size=1, height=1080, width=1920, qp=32, device_id=0):
    """Profile DMCI (I-frame) model"""
    print("=" * 80)
    print(f"Profiling DMCI (I-frame) Model")
    print(f"Input shape: ({batch_size}, 3, {height}, {width})")
    print(f"QP: {qp}")
    print("=" * 80)
    
    with get_accelerator().device(device_id):
        # Initialize model
        model = DMCI()
        model.eval()
        model = model.to(f"cuda:{device_id}")
        model.update()  # Initialize entropy coder
        model.half()  # Convert to half precision
        
        # Create wrapper
        wrapped_model = DMCIWrapper(model)
        
        # Calculate padding
        padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
        padded_height = height + padding_b
        padded_width = width + padding_r
        
        # Create input
        input_shape = (batch_size, 3, padded_height, padded_width)
        dummy_input = torch.randn(input_shape, device=f"cuda:{device_id}", dtype=torch.float16)
        
        # Profile
        flops, macs, params = get_model_profile(
            model=wrapped_model,
            input_shape=input_shape,
            args=(qp,),
            kwargs=None,
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=10,
            warm_up=10,
            as_string=True,
            output_file=None,
            ignore_modules=None
        )
        
        print(f"\nTotal FLOPs: {flops}")
        print(f"Total MACs: {macs}")
        print(f"Total Parameters: {params}")
        print("=" * 80)
        
        return flops, macs, params


def profile_dmc_model(batch_size=1, height=1080, width=1920, qp=32, device_id=0):
    """Profile DMC (P-frame) model"""
    print("=" * 80)
    print(f"Profiling DMC (P-frame) Model")
    print(f"Input shape: ({batch_size}, 3, {height}, {width})")
    print(f"QP: {qp}")
    print("=" * 80)
    
    with get_accelerator().device(device_id):
        # Initialize model
        model = DMC()
        model.eval()
        model = model.to(f"cuda:{device_id}")
        model.update()  # Initialize entropy coder
        model.half()  # Convert to half precision
        
        # Create wrapper
        wrapped_model = DMCWrapper(model)
        
        # Calculate padding
        padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
        padded_height = height + padding_b
        padded_width = width + padding_r
        
        # Create inputs
        input_shape = (batch_size, 3, padded_height, padded_width)
        dummy_x = torch.randn(input_shape, device=f"cuda:{device_id}", dtype=torch.float16)
        
        # Create reference frame (reconstructed frame)
        dummy_recon = torch.randn(input_shape, device=f"cuda:{device_id}", dtype=torch.float16)
        
        # Create reference feature (for P-frame)
        # feature_adaptor_p expects 256 channels (g_ch_d), not 192 channels (g_ch_src_d)
        # This is the processed feature after feature_adaptor_i, not the raw pixel_unshuffle output
        from src.models.video_model import g_ch_d
        feature_shape = (batch_size, g_ch_d, padded_height // 8, padded_width // 8)
        dummy_p_feature = torch.randn(feature_shape, device=f"cuda:{device_id}", dtype=torch.float16)
        
        # Add reference frame to model (needed for apply_feature_adaptor)
        model.add_ref_frame(feature=dummy_p_feature, frame=dummy_recon, increase_poc=False)
        
        # Profile
        flops, macs, params = get_model_profile(
            model=wrapped_model,
            input_shape=input_shape,
            args=(qp,),
            kwargs=None,
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=10,
            warm_up=10,
            as_string=True,
            output_file=None,
            ignore_modules=None
        )
        
        print(f"\nTotal FLOPs: {flops}")
        print(f"Total MACs: {macs}")
        print(f"Total Parameters: {params}")
        print("=" * 80)
        
        return flops, macs, params


def main():
    """Main function to profile both models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile DCVC-RT models using DeepSpeed flops profiler")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--height', type=int, default=1080, help='Frame height')
    parser.add_argument('--width', type=int, default=1920, help='Frame width')
    parser.add_argument('--qp', type=int, default=32, help='Quantization parameter')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--model', type=str, choices=['i', 'p', 'both'], default='both',
                       help='Which model to profile: i (I-frame), p (P-frame), or both')
    
    args = parser.parse_args()
    
    if args.model in ['i', 'both']:
        profile_dmci_model(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            qp=args.qp,
            device_id=args.device_id
        )
    
    if args.model in ['p', 'both']:
        profile_dmc_model(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            qp=args.qp,
            device_id=args.device_id
        )


if __name__ == "__main__":
    main()

