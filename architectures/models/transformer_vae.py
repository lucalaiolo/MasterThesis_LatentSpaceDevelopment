"""Frame-token transformer VAE ([ARCH §4.1, §4.2]).

Each frame is one token. The encoder prepends a class token and reads
the posterior parameters off it. The decoder broadcasts the latent into
T positioned queries and runs a stack of self-attention blocks. An
optional inpainting head concatenates the mask before the output layer,
for Recipe 3.
"""

from __future__ import annotations

from .common import (torch, nn, BottleneckHeads,
                     pack_encoder_input, reparameterise,
                     sinusoidal_positional_encoding)


class TransformerVAE(nn.Module):
    """The frame-token transformer VAE.

    Encoder ([ARCH §4.1]): a linear embedding lifts each frame's 4J
    channels to d_model, sinusoidal positional encoding is added, a
    learnable class token is prepended, and L pre-norm transformer
    blocks with H heads run over the T + 1 tokens. The final class-token
    representation feeds the bottleneck heads.

    Decoder ([ARCH §4.2]): a linear layer lifts the latent to a query
    vector, broadcast to T positions with sinusoidal positional
    encoding, and L pre-norm transformer blocks with H heads run over
    the T tokens. A final linear projection returns 3J channels per
    frame.
    """

    def __init__(self, T: int, J: int, d_z: int = 32,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3,
                 ffn_ratio: int = 4, dropout: float = 0.1,
                 inpainting: bool = False):
        super().__init__()
        self.T = T
        self.J = J
        self.d_z = d_z
        self.d_model = d_model
        self.inpainting = inpainting

        # ---- Encoder pieces ------------------------------------------
        self.token_embed = nn.Linear(4 * J, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer(
            "enc_pos", sinusoidal_positional_encoding(T + 1, d_model),
            persistent=False,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.heads = BottleneckHeads(d_model, d_z)

        # ---- Decoder pieces ------------------------------------------
        self.query_lift = nn.Linear(d_z, d_model)
        self.register_buffer(
            "dec_pos", sinusoidal_positional_encoding(T, d_model),
            persistent=False,
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers)

        # For Recipe 3 the output projection sees J extra channels.
        out_in = d_model + (J if inpainting else 0)
        self.dec_output = nn.Linear(out_in, 3 * J)

        # Small initialisation for the class token, so early training has
        # a well-behaved scale in the attention.
        nn.init.normal_(self.cls_token, std=0.02)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M):
        """Map (clip, mask) to (mu, logvar).

        Args:
            X: (B, T, J, 3).
            M: (B, T, J).
        Returns:
            (mu, logvar), each (B, d_z).
        """
        B = X.shape[0]
        x = pack_encoder_input(X, M)                      # (B, T, 4J)
        tokens = self.token_embed(x)                      # (B, T, d_model)
        cls = self.cls_token.expand(B, 1, self.d_model)   # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, T + 1, d_model)
        tokens = tokens + self.enc_pos.unsqueeze(0)
        out = self.encoder(tokens)                        # (B, T + 1, d_model)
        return self.heads(out[:, 0])                      # class-token slot

    # ---- Decoder ---------------------------------------------------------
    def decode(self, z, M=None):
        """Map z to reconstructed clip.

        Args:
            z: (B, d_z).
            M: (B, T, J) if `inpainting`; ignored otherwise.
        Returns:
            X_hat, (B, T, J, 3).
        """
        B = z.shape[0]
        q = self.query_lift(z)                            # (B, d_model)
        q = q.unsqueeze(1).expand(B, self.T, self.d_model)
        q = q + self.dec_pos.unsqueeze(0)                 # (B, T, d_model)
        out = self.decoder(q)                             # (B, T, d_model)
        if self.inpainting:
            if M is None:
                raise ValueError("Inpainting decoder needs the mask.")
            out = torch.cat([out, M], dim=-1)             # (B, T, d_model + J)
        x_hat = self.dec_output(out)                      # (B, T, 3J)
        return x_hat.reshape(B, self.T, self.J, 3)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M):
        mu, logvar = self.encode(X, M)
        z = reparameterise(mu, logvar)
        X_hat = self.decode(z, M if self.inpainting else None)
        return X_hat, mu, logvar
