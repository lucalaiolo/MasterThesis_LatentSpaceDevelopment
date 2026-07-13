"""Frame-token transformer VAE ([ARCH §4.1, §4.2]).

Each frame is one token. The encoder prepends a class token and reads
the posterior parameters off it. The decoder broadcasts the latent into
T positioned queries and runs a stack of self-attention blocks. For
Recipe 3 the decoder branches into two heads on top of the shared
transformer trunk: a full head and a mask-conditioned inpainting head
scored only on hidden positions ([MVAE §5]).
"""

from __future__ import annotations

from .common import (torch, nn, BottleneckHeads, ConditioningEmbedding,
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
                 inpainting: bool = False,
                 n_cond: int = 0, cond_dim: int = 8,
                 cond_dropout: float = 0.0):
        super().__init__()
        self.T = T
        self.J = J
        self.d_z = d_z
        self.d_model = d_model
        self.inpainting = inpainting

        # Conditioning ([CARE-PD §6]); built only when requested so the
        # plain-VAE parameter budget is untouched.
        self.n_cond = n_cond
        d_c = cond_dim if n_cond > 0 else 0
        self.cond = (ConditioningEmbedding(n_cond, cond_dim, cond_dropout)
                     if n_cond > 0 else None)

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
        # e(c) concatenated to the class-token representation before the
        # posterior heads.
        self.heads = BottleneckHeads(d_model + d_c, d_z)

        # ---- Decoder pieces ------------------------------------------
        # e(c) concatenated to z before the query lift.
        self.query_lift = nn.Linear(d_z + d_c, d_model)
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

        # Full-clip head. Not mask-conditioned; used by all three recipes.
        self.dec_output_full = nn.Linear(d_model, 3 * J)
        # Recipe 3 inpainting head: mask-conditioned ([MVAE §5.1]).
        if inpainting:
            self.dec_output_inp = nn.Linear(d_model + J, 3 * J)

        # Small initialisation for the class token, so early training has
        # a well-behaved scale in the attention.
        nn.init.normal_(self.cls_token, std=0.02)

    # ---- Encoder ---------------------------------------------------------
    def encode(self, X, M, c=None):
        """Map (clip, mask[, cohort]) to (mu, logvar).

        Args:
            X: (B, T, J, 3).
            M: (B, T, J).
            c: optional (B,) conditioning ids; ignored for a plain VAE.
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
        h = out[:, 0]                                     # class-token slot
        if self.cond is not None:
            e = self.cond.encoder_vector(c, B, h.device)
            h = torch.cat([h, e], dim=1)
        return self.heads(h)

    # ---- Decoder ---------------------------------------------------------
    def _decode_trunk(self, z, c=None):
        B = z.shape[0]
        if self.cond is not None:
            e = self.cond.decoder_vector(c, B, z.device, self.training)
            z = torch.cat([z, e], dim=1)
        q = self.query_lift(z)                            # (B, d_model)
        q = q.unsqueeze(1).expand(B, self.T, self.d_model)
        q = q + self.dec_pos.unsqueeze(0)                 # (B, T, d_model)
        return self.decoder(q)                            # (B, T, d_model)

    def decode_full(self, z, c=None):
        """Full-clip reconstruction head, ignoring the mask."""
        B = z.shape[0]
        out = self._decode_trunk(z, c)
        x_hat = self.dec_output_full(out)                 # (B, T, 3J)
        return x_hat.reshape(B, self.T, self.J, 3)

    def decode_inp(self, z, M, c=None):
        """Mask-conditioned inpainting head (Recipe 3 only)."""
        if not self.inpainting:
            raise RuntimeError("Model was built without the inpainting head.")
        B = z.shape[0]
        out = self._decode_trunk(z, c)
        out = torch.cat([out, M], dim=-1)                 # (B, T, d_model + J)
        x_hat = self.dec_output_inp(out)                  # (B, T, 3J)
        return x_hat.reshape(B, self.T, self.J, 3)

    # ---- Combined --------------------------------------------------------
    def forward(self, X, M, c=None):
        """Encode, sample, decode.

        Returns:
            (X_hat_full, mu, logvar) when built without the inpainting head.
            (X_hat_full, X_hat_inp, mu, logvar) for Recipe 3.
        """
        mu, logvar = self.encode(X, M, c)
        z = reparameterise(mu, logvar)
        X_hat_full = self.decode_full(z, c)
        if self.inpainting:
            return X_hat_full, self.decode_inp(z, M, c), mu, logvar
        return X_hat_full, mu, logvar
