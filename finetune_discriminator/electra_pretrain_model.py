from transformers import ElectraConfig,ElectraForMaskedLM
import torch.nn as nn
import torch
import pdb

class ElectraGenerator(nn.Module):

    def __init__(self,config: ElectraConfig,embeddings,generator=None,embed_layer=None, with_cuda = True, pos_embedding = 'emb_dim'):
        super().__init__()
        self.embed_layer = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size, padding_idx= config.vocab_size-1)
        if embed_layer:
            self.embed_layer.load_state_dict(torch.load(embed_layer))
        else:
            self.embed_layer.weight = nn.Parameter(embeddings)        
        if generator:
            self.generator = ElectraForMaskedLM.from_pretrained(generator,config=config)
        else:
            self.generator = ElectraForMaskedLM(config)
        self.softmax = nn.Softmax(dim=2)
        self.sin_emb_layer = PositionalEmbedding(demb=config.embedding_size)
        self.demb = config.embedding_size
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.pos_embedding = pos_embedding

        #basis_func = rbf.gaussian
        #self.rbf_layer = rbf.RBF(1, self.demb, basis_func)

        print("ELECTRA POS embedding:", pos_embedding)


    def forward(self,data,attention_mask, labels, frequencies):
        #pdb.set_trace()
        data = self.embed_layer(data)
        bsz = data.size()[0]
        if not self.pos_embedding in ['absolute', 'relative_key', 'relative_key_query']:
            if self.pos_embedding in ['exact_sine', 'exact_sine_shrink', 'bin_sine']:
                max_frequencies = torch.reshape(torch.max(frequencies, dim=1).values, (-1, 1))
                if self.pos_embedding == 'exact_sine_shrink':
                    frequencies = (1 + frequencies) / (1 + max_frequencies)
                elif self.pos_embedding == 'bin_sine':
                    frequencies = torch.round(torch.log(frequencies + 1)).to(self.device)
                else:
                    frequencies = frequencies + 1
                input_len = data.size()[1]
                for i in range(bsz):
                    pos_embedding_vals = torch.reshape(self.sin_emb_layer(frequencies[i], 1), (input_len, self.demb))
                    data[i] += pos_embedding_vals
            #elif self.pos_embedding == 'rbf':
            #    max_frequencies = torch.reshape(torch.max(frequencies, dim=1).values, (-1, 1))
            #    frequencies / max_frequencies
            #    for i in range(bsz):
            #        pos_embedding_vals = self.rbf_layer(frequencies[i].view(-1, 1))
            #        data[i] += pos_embedding_vals
            elif self.pos_embedding == 'rbf_bin':
                pass
            elif self.pos_embedding == 'emb_dim':
                data[:, :, 0] = torch.log(torch.abs(frequencies) + 1) / 12
            elif self.pos_embedding == 'emb_dim_raw':
                max_frequencies = torch.reshape(torch.max(frequencies, dim=1).values, (-1, 1))
                data[:, :, 0] = frequencies / max_frequencies
            elif self.pos_embedding == 'none':
                pass
        output = self.generator(attention_mask=attention_mask,inputs_embeds=data,labels=labels)
        loss = output.loss
        scores = output.logits
        scores = nn.functional.sigmoid(scores)
        return loss, scores, output.logits



#from: https://huggingface.co/transformers/_modules/transformers/models/transfo_xl/modeling_transfo_xl.html
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        #self.inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        #print(pos_seq.size(), self.inv_freq.size())
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]




