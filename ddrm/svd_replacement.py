import torch
import numpy as np
import torch.fft as tfft



# From https://github.com/bahjat-kawar/ddrm/blob/master/functions/svd_replacement.py
class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        #print('vec', vec.shape)
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))
    
    
class Fourier2D(H_functions):
    def __init__(self, channels, img_dim, S, device):
        self.channels = channels
        self.img_dim = img_dim
        # To make hermitian symmetric
        S = S | S.transpose(0, 1)
        S_flat = S.flatten()
        missing_indices = torch.where(S_flat == 0)[0]
        self.missing_indices = missing_indices

        self.kept_indices = torch.Tensor(
            [i for i in range(channels * img_dim**2) if i not in missing_indices]
        ).to(device).long()

        self._singulars = torch.ones(
            channels * img_dim**2 - missing_indices.shape[0]
        ).to(device)
        
        self.device = device
 

    
    
    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)

        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]

        out = out.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = tfft.ifft2(tfft.ifftshift(out, dim=(-2, -1)), norm='ortho').real
        out = out.reshape(vec.shape[0], -1)

        return out
    
    def Vt(self, vec):
        vec = tfft.fftshift(tfft.fft2(vec.float(), norm='ortho'), dim=(-2,-1))
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out
    
    
    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device, dtype=vec.dtype)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    


# class FourierInpainting(H_functions):
#     def __init__(self, channels, img_dim, missing_indices, device):
#         self.channels = channels
#         self.img_dim = img_dim
#         self.missing_indices = missing_indices
#         self.kept_indices = torch.Tensor(
#             [i for i in range(channels * img_dim**2) if i not in missing_indices]
#         ).to(device).long()
#         self._singulars = torch.ones(
#             channels * img_dim**2 - missing_indices.shape[0]
#         ).to(device)
#         self.device = device

#     def _fft2(self, x):
#         # x: (batch, channels, H, W)
#         return tfft.fft2(x.float())

#     def _ifft2(self, X):
#         # X: (batch, channels, H, W)
#         return tfft.ifft2(X).real

#     def V(self, vec):
#         temp = vec.clone().reshape(vec.shape[0], -1)
#         out = torch.zeros_like(temp)
#         out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
#         out[:, self.missing_indices] = 0.0  
#         out = out.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
#         img = self._ifft2(out)
#         return img.reshape(vec.shape[0], -1)

#     def Vt(self, vec):
#         img = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
#         freq = self._fft2(img)
#         freq_flat = freq.reshape(vec.shape[0], -1)
#         out = torch.zeros_like(freq_flat)
#         out[:, :self.kept_indices.shape[0]] = freq_flat[:, self.kept_indices].real
#         return out

#     def U(self, vec):
#         return vec.clone().reshape(vec.shape[0], -1)

#     def Ut(self, vec):
#         return vec.clone().reshape(vec.shape[0], -1)

#     def singulars(self):
#         return self._singulars

#     def add_zeros(self, vec):
#         temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
#         reshaped = vec.clone().reshape(vec.shape[0], -1)
#         temp[:, :reshaped.shape[1]] = reshaped
#         return temp


