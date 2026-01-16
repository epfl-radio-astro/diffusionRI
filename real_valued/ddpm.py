import math
import torch
import torch.nn as nn
import numpy as np
from helpers import proj_hermitian


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    Extract values from a 1D tensor `a` at indices `t` and reshape for broadcasting.
    a:      (T,)
    t:      (B,) long
    return: (B, 1, 1, 1) or matching x dims
    """
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in the Improved DDPM paper.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.999)


class DDPM(nn.Module):
    def __init__(self, unet, timesteps=200, betas=None, device=torch.device("cuda")):

        super().__init__()
        self.unet = unet
        self.unet = torch.compile(
            self.unet,
            backend="inductor",
            fullgraph=False,
            mode="max-autotune",
            dynamic=False,
        )
        self.device = device

        if betas is None:
            betas = cosine_beta_schedule(timesteps)
        self.betas = torch.from_numpy(betas).float().to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_recip_alphas = 1.0 / torch.sqrt(self.alphas)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.betas = self.betas

        tilde_betas = (
            self.betas
            * (
                1.0
                - torch.cat([self.alpha_bars.new_tensor([1.0]), self.alpha_bars[:-1]])
            )
            / (1.0 - self.alpha_bars)
        )
        self.sqrt_tilde_betas = torch.sqrt(tilde_betas)

        self.num_timesteps = len(self.betas)

    def q_sample(self, x_start, t, noise=None):
        if t.dtype != torch.long or t.dim() != 1:
            raise ValueError(
                "t must be a 1-D Long tensor of length batch_size; "
                f"got shape {t.shape}, dtype {t.dtype}"
            )

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t][
            :, None, None, None
        ]

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample(
        self, x_t: torch.Tensor, t: torch.Tensor, sigma_modifier=1.0
    ) -> torch.Tensor:
        """
        One reverse-diffusion step x_t → x_{t-1}.
        x_t : (B, C, H, W)   float64
        t   : (B,)              long
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_bar = extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        eps_theta = self.unet(x_t, t)

        mean = sqrt_recip_alpha * (x_t - betas_t / sqrt_one_minus_bar * eps_theta)

        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        noise = torch.randn_like(x_t)

        sigma_t = sigma_modifier * extract(self.sqrt_tilde_betas, t, x_t.shape)
        return mean + nonzero_mask * sigma_t * noise

    @torch.no_grad()
    def sample(self, image_size, batch_size, sigma_modifier):
        x = torch.randn((batch_size, 1, image_size, image_size), device=self.device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            x = self.p_sample(x, t_tensor, sigma_modifier=sigma_modifier)
        return x

    def forward(self, x, t):
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        noise_pred = self.unet(x_noisy, t)
        return noise_pred, noise

    @torch.no_grad()
    def sample_show_intermediate(self, image_size, batch_size, sigma_modifier):
        x = torch.randn(batch_size, 1, image_size, image_size, device=self.device)
        ret_list = []
        for t in reversed(range(self.num_timesteps)):
            if t % 200 == 0:
                ret_list.append(x.detach())
            t_tensor = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            x = self.p_sample(x, t_tensor, sigma_modifier=sigma_modifier)
        return x, ret_list

    @torch.no_grad()
    def p_sample_ddrm_loop(self, x, y_0, seq, H_funcs, eta_A, eta_B, eta_C, sigma_0):
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3], device=self.device)
        Sigma[: singulars.shape[0]] = singulars

        U_t_y = H_funcs.Ut(y_0)

        Sig_inv_U_t_y = U_t_y / singulars[: U_t_y.shape[-1]]
        t = torch.full((y_0.shape[0],), seq[-1], dtype=torch.long, device=self.device)

        largest_alphas = extract(self.alpha_bars, t, x.shape)
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()

        large_singulars_index = torch.where(
            singulars * largest_sigmas[0, 0, 0, 0] > sigma_0
        )
        inv_singulars_and_zero = (
            torch.zeros(x.shape[1] * x.shape[2] * x.shape[3])
            .to(singulars.device)
            .to(singulars.dtype)
        )

        inv_singulars_and_zero[large_singulars_index] = (
            sigma_0 / singulars[large_singulars_index]
        )
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

        init_y = (
            torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            .to(x.device)
            .to(U_t_y.dtype)
        )

        init_y[:, large_singulars_index[0]] = U_t_y[
            :, large_singulars_index[0]
        ] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero**2
        remaining_s = (
            remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            .clamp_min(0.0)
            .sqrt()
        )
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size()).to(torch.float32)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.ones(n, dtype=torch.long, device=self.device) * (i)
            next_t = ((torch.ones(n, device=x.device) * (j))).clamp(min=0)


            at = extract(self.alpha_bars, t.long(), xs[-1].shape)
            at_next = extract(self.alpha_bars, next_t.long(), xs[-1].shape)
            xt = xs[-1]

            # consult the model
            et = self.unet(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, : U_t_y.shape[1]]

            falses = torch.zeros(
                V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device
            )
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * eta_C

            sigma_tilde_nextC = torch.sqrt(
                torch.clamp(sigma_next**2 - std_nextC**2, min=0)
            )

            std_nextA = sigma_next * eta_A

            sigma_tilde_nextA = torch.sqrt(
                torch.clamp(sigma_next**2 - std_nextA**2, min=0)
            )

            diff_sigma_t_nextB = torch.sqrt(
                sigma_next**2
                - sigma_0**2 / singulars[cond_before_lite] ** 2 * (eta_B**2)
            )

            # missing pixels
            Vt_xt_mod_next = (
                V_t_x0
                + sigma_tilde_nextC * H_funcs.Vt(et)
                + std_nextC * proj_hermitian(torch.randn_like(V_t_x0) * math.sqrt(2))
            )

            # less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = (
                V_t_x0[:, cond_after]
                + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite]
                + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            )

            # noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = (
                Sig_inv_U_t_y[:, cond_before_lite] * eta_B
                + (1 - eta_B) * V_t_x0[:, cond_before]
                + diff_sigma_t_nextB
                * proj_hermitian(torch.randn_like(U_t_y) * math.sqrt(2))[
                    :, cond_before_lite
                ]
            )

            # aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t)
            xs.append(xt_next)

        xs_ret = [x.to("cpu") for x in xs]
        return xs_ret

    @torch.no_grad()
    def sample_ddrm(
        self, image_size, timesteps, x_0, H_funcs, sigma_0, eta_A, eta_B, eta_C
    ):

        x = torch.randn(x_0.shape[0], 1, image_size, image_size, device=self.device)

        y_0 = H_funcs.H(x_0)
        y_0 = y_0 + sigma_0 * proj_hermitian(torch.randn_like(y_0) * math.sqrt(2))

        pinv_y_0 = H_funcs.H_pinv(y_0)

        # pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

        skip = self.num_timesteps // timesteps
        seq = range(0, self.num_timesteps, skip)
        retval = self.p_sample_ddrm_loop(
            x, y_0, seq, H_funcs, eta_A, eta_B, eta_C, sigma_0 * 2
        )

        return retval, pinv_y_0
