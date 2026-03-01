from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from tqdm import tqdm

from pipelines.Vanilla import Vanilla
        
class BackgroundReconstruction(Vanilla):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_backup = {
            k: v.cpu().clone() for k, v in self.pipe.vae.decoder.state_dict().items()
        }

    def reset_vae(self):
        self.pipe.vae.decoder.load_state_dict(self.vae_backup)
        self.pipe.vae.decoder.to(self.device, dtype=torch.float32)

    def postprocess(self,
                    z_0: torch.Tensor,   # edited image latents [B, C, H, W]
                    x:   torch.Tensor,   # original image [B, C, H, W]
                    m:   torch.Tensor,   # mask [1, H, W]
                ):
        device = self.device
        self.reset_vae()
        vae = self.pipe.vae

        # prepare for optimization
        z_0 = z_0.detach().to(device=device, dtype=torch.float32)
        m = m.detach().to(device=device, dtype=torch.float32).unsqueeze(0)
        x = x.detach().to(device=device, dtype=torch.float32)
        x_hat = self.decode_latents(z_0).to(torch.float32)

        # optimize only necessary parts to save time
        params_to_optimize = list(vae.decoder.conv_out.parameters()) + \
                             list(vae.decoder.mid_block.parameters())
        
        # freeze everything else
        vae.decoder.requires_grad_(False)
        for p in params_to_optimize:
            p.requires_grad = True

        # # optimize decoder
        # self.vae.decoder.requires_grad_(True)
        vae.decoder.train()
        vae.decoder.to(device=device, dtype=torch.float32)

        # init optimizer as stated in the paper
        scaling_factor = vae.config.scaling_factor
        known_region_weight = 100.0
        learning_rate = 0.0001
        num_steps = 20

        optimizer = torch.optim.Adam(
            params_to_optimize, 
            lr=learning_rate
        )

        # dict to track training progress
        history = {"total": [], "masked": [], "unmasked": []}
        plot = False

        with torch.enable_grad():
            for i in tqdm(range(num_steps), desc="Optimizing weights:"):
                optimizer.zero_grad()
                
                decoded = vae.decode(z_0 / scaling_factor, return_dict=False)[0]
                
                # Our mask is inverted. Therefore we invert it in the formulas
                loss_masked = F.mse_loss(decoded * (1 - m), x_hat * (1 - m))
                loss_unmasked = F.mse_loss(decoded * m, x * m)
                
                total_loss = loss_masked + (known_region_weight * loss_unmasked)

                if plot:
                    history["total"].append(total_loss.item())
                    history["masked"].append(loss_masked.item())
                    history["unmasked"].append(loss_unmasked.item())
                
                total_loss.backward()
                optimizer.step()
        
        # just for debugging
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(history["total"], label="Total Loss", color='black', linestyle='--')
            plt.plot(history["masked"], label="Masked Loss", color='blue')
            plt.plot(history["unmasked"], label="Unmasked Loss", color='red')
            plt.yscale('log')
            plt.xlabel("Iteration")
            plt.ylabel("Loss (Log Scale)")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.show()
            
        # toggle evaluation mode back on
        vae.decoder.eval()
        vae.decoder.requires_grad_(False)
        vae.decoder.to(dtype=self.dtype)
