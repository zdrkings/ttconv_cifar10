import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
#from torchsummary import summary
import tensorly as tl
tl.set_backend('pytorch')
import tensornetwork as tn
from tensorly import tt_to_tensor
def tt_to_tensor(tt1, tt2, tt3):
    a1 = tn.Node(tt1)
    b1 = tn.Node(tt2)
    c1 = tn.Node(tt3)
    tn.connect(a1[2], b1[0])
    tn.connect(b1[2], c1[0])

    m = a1 @ b1 @ c1
    # print(m.shape)
    # m1 = tn.contract_between(a1, b1)
    # m = tn.contract_between(m1, c1)
    # m = tn.contract_parallel(e1)

    result = (m).tensor
    # print('shape=', result.shape)
    return result

class ttconv(nn.Module):
    def __init__(self, ranks,  each_core_dim, convnumbers, stride, pad, gi, **kwargs):
        super(ttconv, self).__init__(**kwargs)
        self.convnumbers = convnumbers
        self.ranks = list(ranks)
        self.each_core_dim = list(each_core_dim)
        self.coresnumber = len(self.each_core_dim)

        self.bias = nn.Parameter(torch.zeros(convnumbers), requires_grad=True)
        self.bias.data.uniform_(-0.00001, 0.00001)
        # self.cores = []

        self.stride = stride
        self.pad = pad

        self.cores0 = nn.Parameter(torch.zeros(self.convnumbers, self.each_core_dim[0], self.ranks[0]),
                                   requires_grad=True)
        self.cores1 = nn.Parameter(torch.zeros(self.ranks[0], self.each_core_dim[1], self.ranks[1]), requires_grad=True)
        self.cores2 = nn.Parameter(torch.zeros(self.ranks[1], self.each_core_dim[2]), requires_grad=True)

        self.register_parameter('cores_{}'.format(0), self.cores0)
        self.register_parameter('cores_{}'.format(1), self.cores1)
        self.register_parameter('cores_{}'.format(2), self.cores2)

        if gi == 1:
            init.xavier_uniform_(self.cores0, gain=1e-5)
            init.xavier_uniform_(self.cores1, gain=1e-5)
            init.xavier_uniform_(self.cores2, gain=1e-5)
        else:
            init.xavier_uniform_(self.cores0)
            init.xavier_uniform_(self.cores1)
            init.xavier_uniform_(self.cores2)

    # 下一步定义前向函数
    def forward(self, x):

        # kenerl = tt_to_tensor(self.cores0, self.cores1, self.cores2)
        kenerl1 = torch.tensordot(self.cores0, self.cores1, dims=([2], [0]))
        kenerl = torch.tensordot(kenerl1, self.cores2, dims=([3], [0]))
        outfort = F.conv2d(x, kenerl, self.bias, stride=self.stride, padding=self.pad)

        return outfort

class trans_ttconv(nn.Module):
    def __init__(self, ranks,  each_core_dim, convnumbers, stride, pad, outpad, gi, **kwargs):
        super(trans_ttconv, self).__init__(**kwargs)
        self.convnumbers = convnumbers
        self.ranks = list(ranks)
        self.each_core_dim = list(each_core_dim)
        self.coresnumber = len(self.each_core_dim)

        self.bias = nn.Parameter(torch.zeros(convnumbers), requires_grad=True)
        self.bias.data.uniform_(-0.00001, 0.00001)
        # self.cores = []

        self.stride = stride
        self.pad = pad
        self.outpad = outpad

        self.cores0 = nn.Parameter(torch.zeros(self.convnumbers, self.each_core_dim[0], self.ranks[0]),requires_grad=True)
        self.cores1 = nn.Parameter(torch.zeros(self.ranks[0], self.each_core_dim[1], self.ranks[1]), requires_grad=True)
        self.cores2 = nn.Parameter(torch.zeros(self.ranks[1], self.each_core_dim[2]), requires_grad=True)

        self.register_parameter('cores_{}'.format(0), self.cores0)
        self.register_parameter('cores_{}'.format(1), self.cores1)
        self.register_parameter('cores_{}'.format(2), self.cores2)

        if gi == 1:
            init.xavier_uniform_(self.cores0, gain=1e-5)
            init.xavier_uniform_(self.cores1, gain=1e-5)
            init.xavier_uniform_(self.cores2, gain=1e-5)
        else:
            init.xavier_uniform_(self.cores0)
            init.xavier_uniform_(self.cores1)
            init.xavier_uniform_(self.cores2)

        '''
        self.cores0.data.uniform_(-1.2, 1.2)
        self.cores1.data.uniform_(-1.2, 1.2)
        self.cores2.data.uniform_(-1.2, 1.2)
        '''

    # 下一步定义前向函数
    def forward(self, x):
        # kenerl = tt_to_tensor(self.cores0, self.cores1, self.cores2)
        kenerl1 = torch.tensordot(self.cores0, self.cores1, dims=([2], [0]))
        kenerl = torch.tensordot(kenerl1, self.cores2, dims=([3], [0]))
        outfort = F.conv_transpose2d(x, kenerl, self.bias, stride=self.stride, padding=self.pad,
                                     output_padding=self.outpad)

        return outfort
def drop_connect(x, drop_ratio):
    """
    这个函数在整个Project中都没被用到, 暂时先不考虑它的功能
    """
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    和``Diffusion.Model``中的``TimeEmbedding``一模一样
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    """
    这是一个条件编码模块，将condition编码为embedding
    除了初始化Embedding不同，其他部分与time-embedding无异。
    """
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        # 注意，这里在初始化embedding时有一个细节——``num_embeddings=num_labels+1``也就是10+1=11
        # 本实例中考虑的condition是CIFAR10的label，共10个类别，对应0~9，按理来说只需要10个embedding即可，
        # 但是我们需要给``无条件``情况一个embedding表示，在本实例中就是用``0```来表示，
        # 与此同时10个类别对应的标号分别加一，即1~10(会在``TrainCondition.py``中体现), 因此共需要11个embedding
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, labels):
        cemb = self.condEmbedding(labels)
        return cemb


class DownSample(nn.Module):
    """
    相比于``Diffusion.Model.DownSample``, 这里的降采样模块多加了一个5x5、stride=2的conv层
    前向过程由3x3和5x5卷积输出相加得来，不知为什么这么做，可能为了融合更多尺度的信息
    查看原文(4.Experiments 3~4行)，原文描述所使用的模型与《Diffusion Models Beat GANs on Image Synthesis》所用模型一致，
    但是该文章源码并没有使用这种降采样方式，只是简单的3x3或者avg_pool
    """
    def __init__(self, in_ch):
        super().__init__()
        #self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        #self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)
        self.c1 = ttconv(ranks=(4, 4), each_core_dim=(in_ch, 3, 3), convnumbers=in_ch, stride=2, pad=1, gi=0)
        self.c2 = ttconv(ranks=(4, 4), each_core_dim=(in_ch, 5, 5), convnumbers=in_ch, stride=2, pad=2, gi=0)
    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    """
    相比于``Diffusion.Model.UpSample``, 这里的上采样模块使用反卷积而不是最近邻插值
    同``DownSample``也不明白原因，因该两种方式都可以，看个人喜好。
    """
    def __init__(self, in_ch):
        super().__init__()
        #self.c = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.c = ttconv(ranks=(4, 4), each_core_dim=(in_ch, 3, 3), convnumbers=in_ch, stride=1, pad=1, gi=0)
        #self.t = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.t = trans_ttconv(ranks=(4, 4), each_core_dim=(in_ch, 5, 5), convnumbers=in_ch, stride=2, pad=2,outpad=1, gi=0)
    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):
    """
    和``Diffusion.Model``中的``AttnBlock``一模一样
    """
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """
    相比于``Diffusion.Model.ResBlock``, 这里的残差模块多加了一个条件投射层``self.cond_proj``，
    在这里其实可以直接把它看作另一个time-embedding, 它们参与训练的方式一模一样
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            #nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            ttconv(ranks=(4, 4), each_core_dim=(in_ch, 3, 3), convnumbers=out_ch, stride=1, pad=1, gi=0)
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            #nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            ttconv(ranks=(4, 4), each_core_dim=(out_ch, 3, 3), convnumbers=out_ch, stride=1, pad=1, gi=0)
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, cemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]  # 加上time-embedding
        h += self.cond_proj(cemb)[:, :, None, None]  # 加上conditional-embedding
        h = self.block2(h)                           # 特征融合

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class student_UNet(nn.Module):
    """
    相比于``Diffusion.Model.UNet``, 这里的UNet模块就多加了一个``cond_embedding``，
    还有一个变化是在降采样和上采样阶段没有加自注意力层，只在中间过度的时候加了一次，这我不明白是何用意，
    可能是希望网络不要从自己身上学到太多，多关注condition?(我瞎猜的)
    """
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        #self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.head = ttconv(ranks=(4, 4), each_core_dim=(3, 3, 3), convnumbers=ch, stride=1, pad=1, gi=0)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            #nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
            ttconv(ranks=(4, 4), each_core_dim=(now_ch, 3, 3), convnumbers=3, stride=1, pad=1, gi=0)
        )

    def forward(self, x, t, labels):
        # Timestep embedding
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

if __name__ == '__main__':
    batch_size = 80
    model = student_UNet(
        T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
        num_res_blocks=2, dropout=0.15)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(500, size=[batch_size])
    labels = torch.randint(10, size=[batch_size])
    # resB = ResBlock(128, 256, 64, 0.1)
    # x = torch.randn(batch_size, 128, 32, 32)
    # t = torch.randn(batch_size, 64)
    # labels = torch.randn(batch_size, 64)
    # y = resB(x, t, labels)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.shape)

    print('parasum=', sum(p.numel() for p in model.parameters()))
    y = model(x, t, labels)

    print(y.shape)
    store_path_student = 'ckpt_ranks_4_4_127_.pt'
    student = student_UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
                           num_res_blocks=2, dropout=0.15)
    ckpt = torch.load(
        store_path_student, map_location='cuda:9')
    student.load_state_dict(ckpt)
    student.eval()
    print('finished')