import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, tiv_hid_size, tiv_out_size, tv_hid_size, tv_out_size):
        super().__init__()
        self.linear_tiv = nn.Linear(96, tiv_hid_size)
        self.lstm_tv = nn.LSTM(7727, tv_hid_size, batch_first=True)
        self.flatten = nn.Flatten()

        self.ld1 = nn.Linear(tiv_hid_size, 64)
        self.ld2 = nn.Linear(tv_hid_size * 4, 64)
        self.ld3 = nn.Linear(768, 64)

    def forward(self, tiv, tv, embedding):
        invariant = self.linear_tiv(tiv)
        variant, _ = self.lstm_tv(tv)

        invariant = self.ld1(invariant).unsqueeze(dim=1)
        variant = self.ld2(self.flatten(variant)).unsqueeze(dim=1)
        notes = self.ld3(embedding.to(torch.float)).unsqueeze(dim=1)
        return torch.concat([invariant, variant, notes], dim=1)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, lin_size, label_index, device):
        super().__init__()
        self.task_embedding = nn.Embedding(4, d_model) 
        self.output_embedding = nn.Embedding(len(label_index), d_model)
        self.mask = self._generate_mask(816).to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, lin_size, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, output_indices, task_indices, encoder_output):
        task_embed = self.task_embedding(task_indices.long())
        output_embed = self.output_embedding(output_indices.long())

        q = task_embed + output_embed
        h_dec = self.transformer_decoder(q, encoder_output, tgt_mask=self.mask)
        return h_dec

    def _generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class UniMed(nn.Module):
    def __init__(self, tiv_hid_size, tiv_out_size, tv_hid_size, tv_out_size, num_layers, d_model, num_heads, lin_size, dec_hid_size, code_size, hid_size, label_index, device):
        super().__init__()
        self.encoder = Encoder(tiv_hid_size, tiv_out_size, tv_hid_size, tv_out_size)
        self.decoder = Decoder(num_layers, d_model, num_heads, lin_size, label_index, device)
        self.task_1 = nn.Sequential(nn.Linear(64, hid_size), nn.ReLU(), nn.Linear(hid_size, 1))
        self.task_2 = nn.Sequential(nn.Linear(64, hid_size), nn.ReLU(), nn.Linear(hid_size, 1))
        self.task_3 = nn.Sequential(nn.Linear(64, hid_size), nn.ReLU(), nn.Linear(hid_size, 1))
        self.task_4 = nn.Sequential(nn.Linear(64 * 813, code_size), nn.ReLU(), nn.Linear(code_size, 813))
    def forward(self, tiv, tv, embedding, output_indices, task_indices):
        enc_output = self.encoder(tiv, tv, embedding)
        h_dec = self.decoder(output_indices, task_indices, enc_output)

        shock_output = self.task_1(h_dec[:, 0])
        arf_output = self.task_2(h_dec[:, 1])
        mort_output = self.task_3(h_dec[:, 2])
        code_output = self.task_4(h_dec[:, 3:].view(mort_output.shape[0], -1))

        return (shock_output, arf_output, mort_output, code_output)