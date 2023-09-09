# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


def _shape(x):
    return torch.gather(torch.tensor(x.shape), 0, torch.tensor(
        (0, 1, 2, 3))).tolist()


@FUNCTION_REWRITER.register_rewriter(
    'mmagic.models.editors.deepfillv1.contextual_attention.'
    'ContextualAttentionModule.patch_correlation')
def contextual_attention__patch_correlation(ctx, self, x, kernel):
    # Explicitly force the shape of tensor to avoid the following RuntimeError:
    # Unsupported: ONNX export of convolution for kernel of unknown shape.
    kernel = kernel.reshape(_shape(kernel))
    return ctx.origin_func(self, x, kernel)


@FUNCTION_REWRITER.register_rewriter(
    'mmagic.models.editors.deepfillv1.contextual_attention.'
    'ContextualAttentionModule.patch_copy_deconv')
def contextual_attention__patch_copy_deconv(ctx, self, attention_score,
                                            context_filter):
    # Explicitly force the shape of tensor to avoid the following RuntimeError:
    # Unsupported: ONNX export of convolution for kernel of unknown shape.
    context_filter = context_filter.reshape(_shape(context_filter))
    return ctx.origin_func(self, attention_score, context_filter)