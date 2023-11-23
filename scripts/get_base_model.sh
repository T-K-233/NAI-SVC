# download the base model
wget -P ./logs/44k/ https://huggingface.co/T-K-233/so-vits-svc-dependencies/resolve/main/v410/G_0.pth
wget -P ./logs/44k/ https://huggingface.co/T-K-233/so-vits-svc-dependencies/resolve/main/v410/D_0.pth

# download the base model for diffusion
wget -P ./logs/44k/diffusion/ https://huggingface.co/T-K-233/so-vits-svc-dependencies/resolve/main/v410/model_0.pt
