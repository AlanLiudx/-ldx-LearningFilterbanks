
from typing import Union
import torch
from torch import nn
from torch.nn import functional as F

class SCM(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True):
        """See :func:`compute_scm`."""
        return compute_scm(x, mask=mask, normalize=normalize)


def compute_scm(x: torch.Tensor, mask: torch.Tensor = None, normalize: bool = True):
    """Compute the spatial covariance matrix from a STFT signal x.

    Args:
        x (torch.ComplexTensor): shape  [batch, mics, freqs, frames]
        mask (torch.Tensor): [batch, 1, freqs, frames] or [batch, 1, freqs, frames]. Optional
        normalize (bool): Whether to normalize with the mask mean per bin.

    Returns:
        torch.ComplexTensor, the SCM with shape (batch, mics, mics, freqs)
    """
    batch, mics, freqs, frames = x.shape
    if mask is None:
        mask = torch.ones(batch, 1, freqs, frames)
    if mask.ndim == 3:
        mask = mask[:, None]

    # torch.matmul((mask * x).transpose(1, 2), x.conj().permute(0, 2, 3, 1))
    part1_real=(mask*x).real
    part1_imag=(mask*x).imag
    part2_real=(x.conj()).real
    part2_imag=(x.conj()).imag
    temp_real=torch.einsum("bmft,bnft->bmnf", part1_real,part2_real)-torch.einsum("bmft,bnft->bmnf", part1_imag,part2_imag)
    temp_imag=torch.einsum("bmft,bnft->bmnf", part1_imag,part2_real)+torch.einsum("bmft,bnft->bmnf", part1_real,part2_imag)
    scm=temp_real+1j*temp_imag
    # scm = torch.einsum("bmft,bnft->bmnf", mask * x, x.conj())
    if normalize:
        scm /= mask.sum(-1, keepdim=True).transpose(-1, -2)
    return scm

class Beamformer(nn.Module):
    """Base class for beamforming modules."""

    @staticmethod
    def apply_beamforming_vector(bf_vector: torch.Tensor, mix: torch.Tensor):
        """Apply the beamforming vector to the mixture. Output (batch, freqs, frames).

        Args:
            bf_vector: shape (batch, mics, freqs)
            mix: shape (batch, mics, freqs, frames).
        """
        return torch.einsum("...mf,...mft->...ft", bf_vector.conj(), mix)

    @staticmethod
    def get_reference_mic_vects(
        ref_mic,
        bf_mat: torch.Tensor,
        target_scm: torch.Tensor = None,
        noise_scm: torch.Tensor = None,
    ):
        """Return the reference channel indices over the batch.

        Args:
            ref_mic (Optional[Union[int, torch.Tensor]]): The reference channel.
                If torch.Tensor (ndim>1), return it, it is the reference mic vector,
                If torch.LongTensor of size `batch`, select independent reference mic of the batch.
                If int, select the corresponding reference mic,
                If None, the optimal reference mics are computed with :func:`get_optimal_reference_mic`,
                If None, and either SCM is None, `ref_mic` is set to `0`,
            bf_mat: beamforming matrix of shape (batch, freq, mics, mics).
            target_scm (torch.ComplexTensor): (batch, freqs, mics, mics).
            noise_scm (torch.ComplexTensor): (batch, freqs, mics, mics).

        Returns:
            torch.LongTensor of size ``batch`` to select with the reference channel indices.
        """
        # If ref_mic already has the expected shape.
        if isinstance(ref_mic, torch.Tensor) and ref_mic.ndim > 1:
            return ref_mic

        if (target_scm is None or noise_scm is None) and ref_mic is None:
            ref_mic = 0
        if ref_mic is None:
            batch_mic_idx = get_optimal_reference_mic(
                bf_mat=bf_mat, target_scm=target_scm, noise_scm=noise_scm
            )
        elif isinstance(ref_mic, int):
            batch_mic_idx = torch.LongTensor([ref_mic] * bf_mat.shape[0]).to(bf_mat.device)
        elif isinstance(ref_mic, torch.Tensor):  # Must be 1D
            batch_mic_idx = ref_mic
        else:
            raise ValueError(
                f"Unsupported reference microphone format. Support None, int and 1D "
                f"torch.LongTensor and torch.Tensor, received {type(ref_mic)}."
            )
        # Output (batch, 1, n_mics, 1)
        # import ipdb; ipdb.set_trace()
        ref_mic_vects = F.one_hot(batch_mic_idx, num_classes=bf_mat.shape[-1])[:, None, :, None]
        return ref_mic_vects.to(bf_mat.dtype).to(bf_mat.device)


class RTFMVDRBeamformer(Beamformer):
    def forward(
        self,
        mix: torch.Tensor,
        target_scm: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        r"""Compute and apply MVDR beamformer from the speech and noise SCM matrices.

        :math:`\mathbf{w} =  \displaystyle \frac{\Sigma_{nn}^{-1} \mathbf{a}}{
        \mathbf{a}^H \Sigma_{nn}^{-1} \mathbf{a}}` where :math:`\mathbf{a}` is the
        ATF estimated from the target SCM.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            target_scm (torch.ComplexTensor): (batch, mics, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        # TODO: Implement several RTF estimation strategies, and choose one here, or expose all.
        # Get relative transfer function (1st PCA of Σss)
        e_val, e_vec = torch.symeig(target_scm.permute(0, 3, 1, 2), eigenvectors=True)
        rtf_vect = e_vec[..., -1]  # bfm
        return self.from_rtf_vect(mix=mix, rtf_vec=rtf_vect.transpose(-1, -2), noise_scm=noise_scm)

    def from_rtf_vect(
        self,
        mix: torch.Tensor,
        rtf_vec: torch.Tensor,
        noise_scm: torch.Tensor,
    ):
        """Compute and apply MVDR beamformer from the ATF vector and noise SCM matrix.

        Args:
            mix (torch.ComplexTensor): shape (batch, mics, freqs, frames)
            rtf_vec (torch.ComplexTensor): (batch, mics, freqs)
            noise_scm (torch.ComplexTensor): (batch, mics, mics, freqs)

        Returns:
            Filtered mixture. torch.ComplexTensor (batch, freqs, frames)
        """
        noise_scm_t = noise_scm.permute(0, 3, 1, 2)  # -> bfmm
        rtf_vec_t = rtf_vec.transpose(-1, -2).unsqueeze(-1)  # -> bfm1

        numerator = stable_solve(rtf_vec_t, noise_scm_t)  # -> bfm1

        denominator = torch.matmul(rtf_vec_t.conj().transpose(-1, -2), numerator)  # -> bf11
        bf_vect = (numerator / denominator).squeeze(-1).transpose(-1, -2)  # -> bfm1  -> bmf
        output = self.apply_beamforming_vector(bf_vect, mix=mix)  # -> bft
        return output


if __name__=='__main__':
    x1=torch.randn(8,16,257,251)
    x2=torch.randn(8,16,257,251)
    x=x1+1j*x2
    print("x type:",x.type())
    print("x1 type:",x1.type())
    # mask1=torch.randn(8,16,257,251)
    # mask2=torch.randn(8,16,257,251)
    # mask=mask1+1j*mask2
    mask=torch.randn(8,1,257,251)

    print("x grad fn:",x.grad_fn)
    scm_calculate=SCM()
    x=x.cuda()
    mask=mask.cuda()
    scm_calculate=scm_calculate.cuda()

    output=scm_calculate(x,mask)

    print("success")

