import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


def _estimate_alpha1(feature_reps, targets, device):

    X_T, X = feature_reps, feature_reps.permute(0,2,1)

    try:
        X_TX_inv = torch.linalg.inv(torch.bmm(X_T, X))
    except:
        X_TX_inv = torch.linalg.pinv(torch.bmm(X_T, X))



    X_Ty = torch.bmm(X_T, targets.unsqueeze(-1))

    # Compute likely scores
    alpha_hat = torch.bmm(X_TX_inv, X_Ty)
    return alpha_hat
    #return alpha_hat.squeeze(-1)  # shape (bs,  d) (NOT normalised)

class Delelstm(torch.jit.ScriptModule):
    def __init__(self, Delelstm_params,short):
        super(Delelstm, self).__init__()
        self.input_dim=Delelstm_params['input_dim']
        self.n_units=Delelstm_params['n_units']
        self.depth = Delelstm_params['time_depth']
        self.output_dim = Delelstm_params['output_dim']
        self.N_units = Delelstm_params['N_units']
        self.short=short
    
        #tensor LSTM parameter
        self.U_j = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_i = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_f = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_o = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.W_j = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.B_j = nn.Parameter(torch.randn(self.input_dim, self.n_units))
        self.B_i = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_f = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_o = nn.Parameter(torch.randn(self.input_dim, self.n_units))
  
        #vanilla LSTM parameter
        self.u_j = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.w_j = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.b_j = nn.Parameter(torch.zeros(self.N_units))
        self.w_i = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_i = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_i = nn.Parameter(torch.zeros(self.N_units))
        self.w_f = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_f = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_f = nn.Parameter(torch.zeros(self.N_units))
        self.w_o = nn.Parameter(torch.randn(self.N_units, self.N_units) )
        self.u_o = nn.Parameter(torch.randn(self.input_dim, self.N_units) )
        self.b_o = nn.Parameter(torch.zeros(self.N_units))
        self.w_p = nn.Parameter(torch.randn(self.N_units, 1))
        self.b_p = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x, device):

        H_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        C_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)


        h_tilda_t = torch.zeros(x.shape[0],  self.N_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0],  self.N_units).to(device)
        unorm_list = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])


        for t in range(self.depth-1):
            # eq 1
            temp=H_tilda_t
            J_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.B_j)

            # eq 5
            I_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.B_i)
            F_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.B_f)
            O_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", H_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.B_o)
            # eq 6
            C_tilda_t = C_tilda_t * F_tilda_t + I_tilda_t * J_tilda_t
            # eq 7
            H_tilda_t = (O_tilda_t * torch.tanh(C_tilda_t)) #shape batch, feature, dim

            #normal lstm
            i_t = torch.sigmoid(((x[:, t, :] @ self.u_i) + (h_tilda_t @ self.w_i) + self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.u_f) + (h_tilda_t @ self.w_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.u_o) + (h_tilda_t @ self.w_o) + self.b_o))
            j_t = torch.tanh(((x[:, t, :] @ self.u_j) + (h_tilda_t @ self.w_j) + self.b_j))
            c_tilda_t = f_t * c_tilda_t + i_t * j_t
            h_tilda_t = o_t * torch.tanh(c_tilda_t)
            diff=H_tilda_t-temp
            if (t>self.short):
                newH_tilda_t=torch.concat([temp, diff],dim=1)
                unnorm_weight= _estimate_alpha1(newH_tilda_t, targets=h_tilda_t, device=device)
                h_tilda_t=torch.bmm(newH_tilda_t.permute(0,2,1),unnorm_weight).squeeze(-1)

            #predict
                pred_y=(h_tilda_t @ self.w_p) + self.b_p
                pred_list+=[pred_y]
                unorm_list += [unnorm_weight]
        pred = torch.stack(pred_list).permute(1,0,2)

        unorm = torch.stack(unorm_list) #shape time_depth, BATCH input_dim

        return pred, unorm,None




#baseline model simple LSTM

class normalLSTMpertime(torch.jit.ScriptModule):
    def __init__(self, normalLSTMpertime_params, short):
        super(normalLSTMpertime, self).__init__()
        self.input_dim=normalLSTMpertime_params['input_dim']
        self.n_units=normalLSTMpertime_params['n_units']
        self.depth = normalLSTMpertime_params['time_depth']
        self.output_dim = normalLSTMpertime_params['output_dim']
        
        self.U_j = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.W_j = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.b_j = nn.Parameter(torch.zeros(self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.U_i=nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_i = nn.Parameter(torch.zeros(self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.U_f = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_f = nn.Parameter(torch.zeros(self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.n_units, self.n_units)  )
        self.U_o = nn.Parameter(torch.randn(self.input_dim, self.n_units)  )
        self.b_o = nn.Parameter(torch.zeros(self.n_units))
        self.W_p=nn.Parameter(torch.randn(self.n_units, 1))
        self.b_p= nn.Parameter(torch.zeros(1))
        self.short = short
      

        self.reset_parameters()

    def reset_parameters(self) -> None:
            for name, param in self.named_parameters():
                if 'B' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'U' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'W' in name:
                    nn.init.orthogonal_(param.data)
                elif 'b' in name:
                    nn.init.constant_(param.data, 0.01)
                elif 'u' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'w' in name:
                    nn.init.orthogonal_(param.data)
    def forward(self, x, device):
        h_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0], self.n_units).to(device)
        pred_list = torch.jit.annotate(list[Tensor], [])

        for t in range(self.depth-1):

            i_t=torch.sigmoid(((x[:,t,:]@self.U_i)+(h_tilda_t@self.W_i)+self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.U_f) + (h_tilda_t @ self.W_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.U_o) + (h_tilda_t @ self.W_o) + self.b_o))
            j_t=torch.tanh(((x[:,t,:]@self.U_j)+(h_tilda_t@self.W_j)+self.b_j))
            c_tilda_t=f_t*c_tilda_t+i_t*j_t
            h_tilda_t=o_t*torch.tanh(c_tilda_t)
            if (t>self.short):
                pred_y = (h_tilda_t @ self.W_p) + self.b_p
                pred_list += [pred_y]

        pred = torch.stack(pred_list).permute(1, 0, 2)

        return pred, None, None



class IMVTensorLSTM_pertime(torch.jit.ScriptModule):
    def __init__(self, IMVTensorLSTM_pertime_params, short):
        super(IMVTensorLSTM_pertime, self).__init__()
        self.input_dim = IMVTensorLSTM_pertime_params['input_dim']
        self.n_units = IMVTensorLSTM_pertime_params['n_units']
        self.depth = IMVTensorLSTM_pertime_params['time_depth']
        self.output_dim = IMVTensorLSTM_pertime_params['output_dim']

        self.U_j = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_i = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_f = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.U_o = nn.Parameter(torch.randn( self.input_dim, 1,  self.n_units) )
        self.W_j = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_i = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_f = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.W_o = nn.Parameter(torch.randn( self.input_dim,  self.n_units,  self.n_units) )
        self.b_j = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_i = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_f = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.b_o = nn.Parameter(torch.randn( self.input_dim,  self.n_units) )
        self.F_alpha_n = nn.Parameter(torch.randn( self.input_dim,  self.n_units, 1) )
        self.F_alpha_n_b = nn.Parameter(torch.randn( self.input_dim, 1) )
        self.F_beta = nn.Linear(2 *  self.n_units, 1)
        self.Phi = nn.Linear(2 *  self.n_units, self.output_dim)
        self.short=short
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():

            if 'b' in name  :
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)

            elif 'W' in name:
                nn.init.orthogonal_(param.data)

    #@torch.jit.script_method
    def forward(self, x, device):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        outputs = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        beta_list = torch.jit.annotate(list[Tensor], [])
        alpha_list = torch.jit.annotate(list[Tensor], [])

        for t in range(x.shape[1]-1):
            outputs += [h_tilda_t]
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            if (t> self.short):
                newoutputs = torch.stack(outputs)
                newoutputs = newoutputs.permute(1, 0, 2, 3)
                # eq 8
                alphas = torch.tanh(torch.einsum("btij,ijk->btik", newoutputs, self.F_alpha_n) + self.F_alpha_n_b)
                alphas = torch.exp(alphas)
                alphas = alphas / torch.sum(alphas, dim=1, keepdim=True) #alphas  shape batch, time, feature, 1
                g_n = torch.sum(alphas * newoutputs, dim=1)
                hg = torch.cat([g_n, h_tilda_t], dim=2)
                mu = self.Phi(hg)
                betas = torch.tanh(self.F_beta(hg)) #betas shape batch, feature,1
                betas = torch.exp(betas)
                betas = betas / torch.sum(betas, dim=1, keepdim=True)
                mean = torch.sum(betas * mu, dim=1)  #mean shape batch 1
                pred_list += [mean]
                alpha_list+=[alphas]
                beta_list+=[betas]

        pred = torch.stack(pred_list).permute(1, 0, 2) #pred shape batch, tiem depth, 1
        alpha_list=torch.cat(alpha_list, dim=1).squeeze(-1)#alphas_list shape batch, 所有的时间长度，feature， 1
        beta_list=torch.stack(beta_list).squeeze(-1).permute(1,0,2)#beta list shape  batch, tiem depth,feature

        return pred, alpha_list, beta_list


class IMVFullLSTM_pertime(torch.jit.ScriptModule):
    def __init__(self, IMVFullLSTM_pertime_params, short):
        super(IMVFullLSTM_pertime, self).__init__()
        self.input_dim=IMVFullLSTM_pertime_params['input_dim']
        self.n_units=IMVFullLSTM_pertime_params['n_units']
        self.depth = IMVFullLSTM_pertime_params['time_depth']
        self.output_dim = IMVFullLSTM_pertime_params['output_dim']

        self.U_j = nn.Parameter(torch.randn(self.input_dim, 1,  self.n_units) )
        self.W_j = nn.Parameter(torch.randn(self.input_dim,  self.n_units,  self.n_units) )
        self.b_j = nn.Parameter(torch.randn(self.input_dim,  self.n_units) )
        self.W_i = nn.Linear( self.input_dim*( self.n_units+1),  self.input_dim* self.n_units)
        self.W_f = nn.Linear( self.input_dim*( self.n_units+1),  self.input_dim* self.n_units)
        self.W_o = nn.Linear( self.input_dim*( self.n_units+1),  self.input_dim* self.n_units)
        self.F_alpha_n = nn.Parameter(torch.randn( self.input_dim,  self.n_units, 1) )
        self.F_alpha_n_b = nn.Parameter(torch.randn( self.input_dim, 1) )
        self.F_beta = nn.Linear(2* self.n_units, 1)
        self.Phi = nn.Linear(2* self.n_units, self.output_dim)
        self.short=short
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)

    #@torch.jit.script_method
    def forward(self, x, device):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).to(device)
        outputs = torch.jit.annotate(list[Tensor], [])
        pred_list = torch.jit.annotate(list[Tensor], [])
        beta_list = torch.jit.annotate(list[Tensor], [])
        alpha_list = torch.jit.annotate(list[Tensor], [])
        for t in range(x.shape[1]-1):
            outputs += [h_tilda_t]
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.reshape(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).reshape(h_tilda_t.shape[0], self.input_dim, self.n_units)
            if(t>self.short):
                newoutputs = torch.stack(outputs)
                newoutputs = newoutputs.permute(1, 0, 2, 3)
            # eq 8
                alphas = torch.tanh(torch.einsum("btij,ijk->btik", newoutputs, self.F_alpha_n) +self.F_alpha_n_b)
                alphas = torch.exp(alphas)
                alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
                g_n = torch.sum(alphas*newoutputs, dim=1)
                hg = torch.cat([g_n, h_tilda_t], dim=2)
                mu = self.Phi(hg)
                betas = torch.tanh(self.F_beta(hg))
                betas = torch.exp(betas)
                betas = betas/torch.sum(betas, dim=1, keepdim=True)
                mean = torch.sum(betas*mu, dim=1)
                pred_list += [mean]
                alpha_list += [alphas]
                beta_list += [betas]

        pred = torch.stack(pred_list).permute(1, 0, 2)
        alpha_list = torch.cat(alpha_list, dim=1).squeeze(-1)  # alphas_list shape batch, 所有的时间长度，feature， 1
        beta_list = torch.stack(beta_list).squeeze(-1).permute(1, 0, 2)  # beta list shape  batch, tiem depth,feature

        return pred, alpha_list, beta_list


class Retain_pertime(nn.Module):
    def __init__(self, Retain_pertime_params, short):
        super(Retain_pertime, self).__init__()
        self.inputDimSize = Retain_pertime_params['input_dim']
        self.embDimSize =Retain_pertime_params['embDimSize']
        self.alphaHiddenDimSize = Retain_pertime_params['alphaHiddenDimSize']
        self.betaHiddenDimSize = Retain_pertime_params['betaHiddenDimSize']
        self.outputDimSize = Retain_pertime_params['output_dim']
        self.keep_prob = Retain_pertime_params['keep_prob']
        self.embedding = nn.Linear(self.inputDimSize, self.embDimSize)
        self.dropout = nn.Dropout(self.keep_prob)
        self.gru_alpha = nn.GRU(self.embDimSize, self.alphaHiddenDimSize)
        self.gru_beta = nn.GRU(self.embDimSize, self.betaHiddenDimSize)
        self.alpha_att = nn.Linear(self.alphaHiddenDimSize, 1)
        self.beta_att = nn.Linear(self.betaHiddenDimSize, self.embDimSize)
        self.out = nn.Linear(self.embDimSize, self.outputDimSize)
        self.short=short

    def initHidden_alpha(self, batch_size,device):
        return torch.zeros(1, batch_size, self.alphaHiddenDimSize).to(device)

    def initHidden_beta(self, batch_size, device):
        return torch.zeros(1, batch_size, self.betaHiddenDimSize).to(device)

    def attentionStep(self, h_a, h_b, att_timesteps):
        """
        两个attention的处理，其中att_timesteps是目前为止的步数
        返回的是一个3维向量，维度为(n_timesteps × n_samples × embDimSize)
        :param h_a:
        :param h_b:
        :param att_timesteps:
        :return:
        """
        reverse_emb_t = self.emb[:att_timesteps].flip(dims=[0]) #在时间维度翻转 time, batch,embedding size
        reverse_h_a = self.gru_alpha(reverse_emb_t, h_a)[0].flip(dims=[0]) * 0.5 #shape time，batch， alphadimension
        reverse_h_b = self.gru_beta(reverse_emb_t, h_b)[0].flip(dims=[0]) * 0.5 #shape time，batch， alphadimension

        preAlpha = self.alpha_att(reverse_h_a)  #time，batch
        preAlpha = torch.squeeze(preAlpha, dim=2)
        alpha = torch.transpose(F.softmax(torch.transpose(preAlpha, 0, 1)), 0, 1) #维度 time,128
        beta = torch.tanh(self.beta_att(reverse_h_b)) #time,batch,emb

        c_t = torch.sum((alpha.unsqueeze(2) * beta * self.emb[:att_timesteps]), dim=0) #batch,emb
        return c_t, alpha, beta

    def forward(self, x, device):
        temp=x.permute(1,0,2)
        first_h_a = self.initHidden_alpha(temp.shape[1], device)
        first_h_b = self.initHidden_beta(temp.shape[1], device)

        self.emb = self.embedding(temp)
        w_emb=self.embedding.weight.data
        if self.keep_prob < 1:
            self.emb = self.dropout(self.emb)

        count = np.arange(temp.shape[0]-1)+1
        pred_list = torch.jit.annotate(list[Tensor], [])
        weight_list = torch.jit.annotate(list[Tensor], [])
         # shape=(seq_len, batch_size, day_dim)
        for i, att_timesteps in enumerate(count):

             c_t, alpha, beta = self.attentionStep(first_h_a, first_h_b, att_timesteps)
             if(i>self.short):
                 y_hat=self.out(c_t) #shape batch， 1
                 w_out=self.out.weight.data #shape outputdim, embdim
                 pred_list += [y_hat]

                #feature weight
                 new_beta = beta.permute(1, 0, 2).unsqueeze(-1)
                 d = torch.mul(new_beta, w_emb)
                 e = torch.matmul(w_out, d) .squeeze(2)
                 new_alpha = alpha.permute(1, 0).unsqueeze(-1)
                 f = torch.mul(new_alpha, e)

                 f=torch.mul(f,x[:,:att_timesteps,:])
                 g = torch.mean(f, dim=1)
                 weight_list +=[g]


        pred = torch.stack(pred_list).permute(1, 0, 2)
        weight_list=torch.stack(weight_list).permute(1,0,2)

        return pred, weight_list,None


class MultiHeadCrossVariableAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_threshold: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_threshold = attn_threshold
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_t: Tensor):
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(0)
        bsz, d, h = h_t.shape
        qkv = self.qkv(h_t)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(bsz, d, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, d, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, d, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(attn >= self.attn_threshold, attn, torch.zeros_like(attn))

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, d, self.hidden_dim)
        z = self.out_proj(context)
        a_t = attn.mean(dim=1)
        return z, a_t


class InteractionDecomposition(nn.Module):
    def __init__(self, hidden_dim: int, ridge_lambda: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ridge_lambda = ridge_lambda
        # Old: self.psi_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # New: Add bias back for better expressivity on real data
        # While bias=False is theoretically cleaner for pure multiplication,
        # real data often has shifts that bias helps model.
        # We keep the multiplicative interaction structure (z_i * z_j)
        self.psi_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, h_prev: Tensor, delta_h: Tensor, z: Tensor, a_t: Tensor, h_t: Tensor):
        bsz, d, h = h_prev.shape
        device = h_prev.device
        
        # Vectorized implementation
        # 1. Determine active interactions (union of masks across batch)
        eye = torch.eye(d, device=device, dtype=torch.bool)
        # a_t shape: (bsz, d, d)
        # mask shape: (bsz, d, d)
        mask = (a_t > 0) & (~eye)
        # Union mask: if pair (i, j) is active in ANY sample, keep it
        union_mask = mask.any(dim=0) # shape (d, d)
        idx = union_mask.nonzero(as_tuple=False) # shape (K, 2)
        
        # 2. Construct Phi matrix
        # phi_long: (bsz, h, d)
        phi_long = h_prev.transpose(1, 2)
        # phi_short: (bsz, h, d)
        phi_short = delta_h.transpose(1, 2)
        
        if idx.numel() == 0:
            phi_inter = torch.empty(bsz, h, 0, device=device, dtype=h_prev.dtype)
        else:
            i_idx = idx[:, 0]
            j_idx = idx[:, 1]
            # Gather z for all samples at once
            # z shape: (bsz, d, h)
            # z[:, i_idx, :] shape: (bsz, K, h)
            
            # Explicit Element-wise Product (Hadamard Product)
            # This forces the model to use multiplicative interaction as the basis
            pair = z[:, i_idx, :] * z[:, j_idx, :] # (bsz, K, h)
            
            psi = torch.tanh(self.psi_proj(pair)) # (bsz, K, h)
            
            # Gather weights
            # a_t[:, i_idx, j_idx] shape: (bsz, K)
            w = a_t[:, i_idx, j_idx].unsqueeze(-1) # (bsz, K, 1)
            psi = psi * w
            phi_inter = psi.transpose(1, 2) # (bsz, h, K)

        phi = torch.cat([phi_long, phi_short, phi_inter], dim=2) # (bsz, h, 2d + K)
        
        # 3. Ridge Regression
        # y = h_t.unsqueeze(-1) # (bsz, h, 1)
        y = h_t.unsqueeze(-1)
        
        # A = phi.mT @ phi -> (bsz, M, M)
        a_mat = torch.bmm(phi.transpose(1, 2), phi)
        # B = phi.mT @ y -> (bsz, M, 1)
        b_vec = torch.bmm(phi.transpose(1, 2), y)
        
        k_dim = phi.shape[2]
        reg = self.ridge_lambda * torch.eye(k_dim, device=device, dtype=phi.dtype).unsqueeze(0) # (1, M, M)
        
        # Batch solve
        # A + regI: (bsz, M, M)
        try:
            theta = torch.linalg.solve(a_mat + reg, b_vec)
        except Exception:
            # Fallback to pseudoinverse if singular
            theta = torch.matmul(torch.linalg.pinv(a_mat + reg), b_vec)
            
        # theta shape: (bsz, M, 1)
        
        # 4. Compute h_hat
        # phi @ theta -> (bsz, h, M) @ (bsz, M, 1) -> (bsz, h, 1)
        h_hat = torch.bmm(phi, theta).squeeze(-1)
        
        # 5. Extract alpha, beta, gamma
        # theta is (bsz, 2d+K, 1). Squeeze last dim -> (bsz, 2d+K)
        theta_flat = theta.squeeze(-1)
        
        alpha = theta_flat[:, :d]
        beta = theta_flat[:, d:2*d]
        gamma_vals = theta_flat[:, 2*d:] # (bsz, K)
        
        gamma_mat = torch.zeros(bsz, d, d, device=device, dtype=h_prev.dtype)
        if idx.numel() != 0:
            # Scatter gamma values back to matrix
            gamma_mat[:, i_idx, j_idx] = gamma_vals

        return h_hat, alpha, beta, gamma_mat


class AttentionDeLELSTM(nn.Module):
    def __init__(self, params, short):
        super().__init__()
        self.input_dim = params['input_dim']
        self.n_units = params['n_units']
        self.depth = params['time_depth']
        self.output_dim = params['output_dim']
        self.N_units = params['N_units']
        self.short = short

        self.attention_heads = params['attention_heads']
        self.attention_threshold = params['attention_threshold']
        self.ridge_lambda = params['ridge_lambda']

        self.U_j = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_i = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_f = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_o = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.W_j = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.B_j = nn.Parameter(torch.randn(self.input_dim, self.n_units))
        self.B_i = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_f = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_o = nn.Parameter(torch.randn(self.input_dim, self.n_units))

        self.u_j = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.w_j = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.b_j = nn.Parameter(torch.zeros(self.N_units))
        self.w_i = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_i = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_i = nn.Parameter(torch.zeros(self.N_units))
        self.w_f = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_f = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_f = nn.Parameter(torch.zeros(self.N_units))
        self.w_o = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_o = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_o = nn.Parameter(torch.zeros(self.N_units))

        self.attn = MultiHeadCrossVariableAttention(self.n_units, self.attention_heads, self.attention_threshold)
        self.decomp = InteractionDecomposition(self.n_units, self.ridge_lambda)

        self.w_p = nn.Parameter(torch.randn(self.n_units, 1))
        self.b_p = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                nn.init.orthogonal_(param.data)
        
        # Explicit initialization for psi_proj to avoid orthogonal init on non-square matrix
        if hasattr(self, 'decomp') and hasattr(self.decomp, 'psi_proj'):
             nn.init.xavier_uniform_(self.decomp.psi_proj.weight)

    def forward(self, x, device, return_attention: bool = False):
        h_var = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        c_var = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        h_mix = torch.zeros(x.shape[0], self.N_units).to(device)
        c_mix = torch.zeros(x.shape[0], self.N_units).to(device)

        pred_list = []
        alpha_list = []
        beta_list = []
        gamma_list = []
        a_list = []

        for t in range(self.depth - 1):
            h_prev = h_var
            j_var = torch.tanh(torch.einsum("bij,ijk->bik", h_var, self.W_j) +
                               torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.B_j)
            i_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_i) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.B_i)
            f_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_f) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.B_f)
            o_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_o) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.B_o)
            c_var = c_var * f_var + i_var * j_var
            h_var = o_var * torch.tanh(c_var)
            delta_h = h_var - h_prev

            i_t = torch.sigmoid(((x[:, t, :] @ self.u_i) + (h_mix @ self.w_i) + self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.u_f) + (h_mix @ self.w_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.u_o) + (h_mix @ self.w_o) + self.b_o))
            j_t = torch.tanh(((x[:, t, :] @ self.u_j) + (h_mix @ self.w_j) + self.b_j))
            c_mix = f_t * c_mix + i_t * j_t
            h_mix = o_t * torch.tanh(c_mix)

            if t > self.short:
                z, a_t = self.attn(h_var)
                h_hat, alpha, beta, gamma = self.decomp(h_prev, delta_h, z, a_t, h_mix)
                
                # Closed loop: Use decomposed state for the next time step
                h_mix = h_hat
                
                pred_y = (h_hat @ self.w_p) + self.b_p
                pred_list.append(pred_y)
                alpha_list.append(alpha)
                beta_list.append(beta)
                gamma_list.append(gamma)
                if return_attention:
                    a_list.append(a_t)

        pred = torch.stack(pred_list, dim=1)
        alpha = torch.stack(alpha_list, dim=0)
        beta = torch.stack(beta_list, dim=0)
        gamma = torch.stack(gamma_list, dim=0)
        if return_attention:
            a_seq = torch.stack(a_list, dim=0)
            return pred, alpha, beta, gamma, a_seq
        return pred, alpha, beta, gamma


class DeLELSTM_AttnNoDecomp(nn.Module):
    def __init__(self, params, short):
        super().__init__()
        self.input_dim = params['input_dim']
        self.n_units = params['n_units']
        self.depth = params['time_depth']
        self.output_dim = params['output_dim']
        self.N_units = params['N_units']
        self.short = short

        self.attention_heads = params['attention_heads']
        self.attention_threshold = params['attention_threshold']

        self.U_j = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_i = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_f = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.U_o = nn.Parameter(torch.randn(self.input_dim, 1, self.n_units))
        self.W_j = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_i = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_f = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.W_o = nn.Parameter(torch.randn(self.input_dim, self.n_units, self.n_units))
        self.B_j = nn.Parameter(torch.randn(self.input_dim, self.n_units))
        self.B_i = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_f = nn.Parameter(torch.Tensor(self.input_dim, self.n_units))
        self.B_o = nn.Parameter(torch.randn(self.input_dim, self.n_units))

        self.u_j = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.w_j = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.b_j = nn.Parameter(torch.zeros(self.N_units))
        self.w_i = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_i = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_i = nn.Parameter(torch.zeros(self.N_units))
        self.w_f = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_f = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_f = nn.Parameter(torch.zeros(self.N_units))
        self.w_o = nn.Parameter(torch.randn(self.N_units, self.N_units))
        self.u_o = nn.Parameter(torch.randn(self.input_dim, self.N_units))
        self.b_o = nn.Parameter(torch.zeros(self.N_units))

        self.attn = MultiHeadCrossVariableAttention(self.n_units, self.attention_heads, self.attention_threshold)
        self.h_proj = nn.Linear(self.n_units, self.N_units, bias=False)

        self.w_p = nn.Parameter(torch.randn(self.N_units, 1))
        self.b_p = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'B' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'U' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'W' in name:
                nn.init.orthogonal_(param.data)
            elif 'b' in name:
                nn.init.constant_(param.data, 0.01)
            elif 'u' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'w' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x, device):
        h_var = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        c_var = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device)
        h_mix = torch.zeros(x.shape[0], self.N_units).to(device)
        c_mix = torch.zeros(x.shape[0], self.N_units).to(device)

        pred_list = []

        for t in range(self.depth - 1):
            j_var = torch.tanh(torch.einsum("bij,ijk->bik", h_var, self.W_j) +
                               torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.B_j)
            i_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_i) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.B_i)
            f_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_f) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.B_f)
            o_var = torch.sigmoid(torch.einsum("bij,ijk->bik", h_var, self.W_o) +
                                  torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.B_o)
            c_var = c_var * f_var + i_var * j_var
            h_var = o_var * torch.tanh(c_var)

            i_t = torch.sigmoid(((x[:, t, :] @ self.u_i) + (h_mix @ self.w_i) + self.b_i))
            f_t = torch.sigmoid(((x[:, t, :] @ self.u_f) + (h_mix @ self.w_f) + self.b_f))
            o_t = torch.sigmoid(((x[:, t, :] @ self.u_o) + (h_mix @ self.w_o) + self.b_o))
            j_t = torch.tanh(((x[:, t, :] @ self.u_j) + (h_mix @ self.w_j) + self.b_j))
            c_mix = f_t * c_mix + i_t * j_t
            h_mix = o_t * torch.tanh(c_mix)

            if t > self.short:
                z, _ = self.attn(h_var)
                h_att = z.mean(dim=1)
                h_hat = self.h_proj(h_att)
                h_mix = h_hat
                pred_y = (h_hat @ self.w_p) + self.b_p
                pred_list.append(pred_y)

        pred = torch.stack(pred_list, dim=1)
        return pred, None, None

