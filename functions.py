import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        # inputs: [2,3,3,4], codebook: [8,4] (K=8 embeddings, each of D=4)
        with torch.no_grad():
            embedding_size = codebook.size(1) #=4
            inputs_size = inputs.size() #= [2,3,3,4]
            inputs_flatten = inputs.view(-1, embedding_size) #= [18,4]

            codebook_sqr = torch.sum(codebook ** 2, dim=1) # squared L2 norms of each vector in the codebook, size [8]
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True) # squared L2 norms of each vector in the inputs, size [18,1]

            # Compute the distances to the codebook
            # squred distance between each flattened input and each codebook vector
            # for each input vector x_i and codebook vector c_j, the distance is: ||x_i - c_j||² = ||x_i||² + ||c_j||² - 2·(x_i·c_j)
            # output size: [18,8]
            # Each row represents one flattened input vector
            # Each column represents its distance to one of the 8 codebook vectors
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            # for each 18 inputs, find the index of the codebook vector with the smallest distance
            # indices_flatten size: [18], values: [2,5,0,...] (values between 0 and 7)
            _, indices_flatten = torch.min(distances, dim=1)
            
            # This reshapes the flattened indices back to match the spatial dimensions of the input:
            # inputs_size[:-1] size: [2, 3, 3] (batch, height, width)
            #The resulting indices tensor has one index per spatial location in each input
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)
            
            '''
            indices: [
                        [ [2, 5, 1],
                            [0, 3, 7],
                            [4, 6, 2] ],
                            
                        [ [1, 0, 4],
                            [7, 5, 3],
                            [2, 6, 1] ]
                        ]
            And the codebook is a tensor of shape [8, 4], then:
            At position [0,0,0], we have index 2, which points to codebook[2] (a 4-dimensional vector)
            At position [0,1,2], we have index 7, which points to codebook[7] (a 4-dimensional vector)
            
            '''
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]