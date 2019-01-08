from collections import defaultdict
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from gcn.modules import fgl
import data.constants as constants
import data.ward_tree as ward_tree


class RClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        self.args = args
        meta = self.args.meta
        wtree = copy.deepcopy(args.wtree)
        wtree.make_val2region(args.nregions)
        self.z_size = z_size

        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size * 32, z_size * 8, z_size]
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        list_of_dict_adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = wtree.region_faithful_go_up_to_reduce(cur_level, next_count)
            list_of_dict_adj_list.append(adj)

        self.downsample0 = fgl.RegionFGL(int(self.channel_sizes[0]), int(self.node_sizes[0]), int(self.channel_sizes[1]), int(self.node_sizes[1]), list_of_dict_adj_list[0], reduction='sum')
        self.downsample1 = fgl.RegionFGL(int(self.channel_sizes[1]), int(self.node_sizes[1]), int(self.channel_sizes[2]), int(self.node_sizes[2]), list_of_dict_adj_list[1], reduction='sum')
        self.downsample2 = fgl.RegionFGL(int(self.channel_sizes[2]), int(self.node_sizes[2]), int(self.channel_sizes[3]), int(self.node_sizes[3]), list_of_dict_adj_list[2], reduction='sum')
        self.downsample3 = fgl.RegionFGL(int(self.channel_sizes[3]), int(self.node_sizes[3]), int(self.channel_sizes[4]), int(self.node_sizes[4]), list_of_dict_adj_list[3], reduction='sum')
        self.downsample4 = fgl.RegionFGL(int(self.channel_sizes[4]), int(self.node_sizes[4]), int(self.channel_sizes[5]), int(self.node_sizes[5]), list_of_dict_adj_list[4], reduction='sum')

        self.activation0 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(p=0.5))
        self.activation1 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(p=0.5))
        self.activation2 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(p=0.5))
        self.activation3 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(p=0.5))
        self.activation4 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(p=0.5))

        self.fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(5):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


class Classifier1(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        # super(Classifier0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size

        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        # self.node_sizes = [in_features, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        # self.node_sizes = [in_features, 1024, 256, 128]
        # self.channel_sizes = [1, 16, 32, 64]  # That mapping should be fairly fast
        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size * 32, z_size * 8, z_size]
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from ~200k->...->128
        # we need to transpose each one and them reverse the list
        self.n_layers = len(self.channel_sizes) - 1
        self.downsample0 = fgl.FGL(int(self.channel_sizes[0]), int(self.node_sizes[0]), int(self.channel_sizes[1]), int(self.node_sizes[1]), adj_list[0])
        self.downsample1 = fgl.FGL(int(self.channel_sizes[1]), int(self.node_sizes[1]), int(self.channel_sizes[2]), int(self.node_sizes[2]), adj_list[1])
        self.downsample2 = fgl.FGL(int(self.channel_sizes[2]), int(self.node_sizes[2]), int(self.channel_sizes[3]), int(self.node_sizes[3]), adj_list[2])
        self.downsample3 = fgl.FGL(int(self.channel_sizes[3]), int(self.node_sizes[3]), int(self.channel_sizes[4]), int(self.node_sizes[4]), adj_list[3])
        self.downsample4 = fgl.FGL(int(self.channel_sizes[4]), int(self.node_sizes[4]), int(self.channel_sizes[5]), int(self.node_sizes[5]), adj_list[4])

        self.activation0 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation1 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation2 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation3 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation4 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))

        self.fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(self.n_layers):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


class Classifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        super().__init__()
        # super(Classifier0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size

        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        # self.node_sizes = [in_features, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        self.node_sizes = [in_features, z_size * 512, z_size * 128, z_size * 32, z_size * 8, z_size]
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.node_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from ~200k->...->128
        # we need to transpose each one and them reverse the list

        self.downsample0 = fgl.FGL_useless(int(self.channel_sizes[0]), int(self.node_sizes[0]), int(self.channel_sizes[1]), int(self.node_sizes[1]), adj_list[0])
        self.downsample1 = fgl.FGL_useless(int(self.channel_sizes[1]), int(self.node_sizes[1]), int(self.channel_sizes[2]), int(self.node_sizes[2]), adj_list[1])
        self.downsample2 = fgl.FGL_useless(int(self.channel_sizes[2]), int(self.node_sizes[2]), int(self.channel_sizes[3]), int(self.node_sizes[3]), adj_list[2])
        self.downsample3 = fgl.FGL_useless(int(self.channel_sizes[3]), int(self.node_sizes[3]), int(self.channel_sizes[4]), int(self.node_sizes[4]), adj_list[3])
        self.downsample4 = fgl.FGL_useless(int(self.channel_sizes[4]), int(self.node_sizes[4]), int(self.channel_sizes[5]), int(self.node_sizes[5]), adj_list[4])

        self.activation0 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation1 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation2 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation3 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.activation4 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.5))

        self.fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], self.channel_sizes[-1]),
            nn.Linear(self.channel_sizes[-1], len(meta['c2i'])),
        )

    def forward(self, x):
        # x: N, constants.masked_nnz
        N = x.shape[0]
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        for i in range(5):
            cur_z = getattr(self, 'downsample{}'.format(i))(cur_z)
            cur_z = getattr(self, 'activation{}'.format(i))(cur_z)
        return self.fc(cur_z.view(N, -1))


"""
DEAD ZONE UNDERNEATH. DO NOT GO BELOW THIS LINE
                      :::!~!!!!!:.
                  .xUHWH!! !!?M88WHX:.
                .X*#M@$!!  !X!M$$$$$$WWx:.
               :!!!!!!?H! :!$!$$$$$$$$$$8X:
              !!~  ~:~!! :~!$!#$$$$$$$$$$8X:
             :!~::!H!<   ~.U$X!?R$$$$$$$$MM!
             ~!~!!!!~~ .:XW$$$U!!?$$$$$$RMM!
               !:~~~ .:!M"T#$$$$WX??#MRRMMM!
               ~?WuxiW*`   `"#$$$$8!!!!??!!!
             :X- M$$$$       `"T#$T~!8$WUXU~
            :%`  ~#$$$m:        ~!~ ?$$$$$$
          :!`.-   ~T$$$$8xx.  .xWW- ~""##*"
.....   -~~:<` !    ~?T#$$@@W@*?$$      /`
W$@@M!!! .!~~ !!     .:XUW$W!~ `"~:    :
#"~~`.:x%`!!  !H:   !WM$$$$Ti.: .!WUn+!`
:::~:!!`:X~ .: ?H.!u "$$$B$$$!W:U!T$$M~
.~~   :X@!.-~   ?@WTWo("*$$$W$TH$! `
Wi.~!X$?!-~    : ?$$$B$Wu("**$RM!
$R@i.~~ !     :   ~$$$$$B$$en:``
?MXT@Wx.~    :     ~"##*$$$$M~
"""
class HierarchicalClassifier0(nn.Module):
    def __init__(self, args, loadable_state_dict=None, z_size=128, dropout_rate=0.5, downsampled=False):
        raise NotImplementedError()
        super(HierarchicalClassifier0, self).__init__()
        self.args = args
        meta = self.args.meta
        wtree = args.wtree
        self.z_size = z_size

        if downsampled:
            in_features = constants.downsampled_masked_nnz
        else:
            in_features = constants.original_masked_nnz
        self.node_sizes = [in_features, z_size * 256, z_size * 64, z_size * 16, z_size * 4, z_size]
        self.channel_sizes = [1, z_size // 16, z_size // 8, z_size // 4, z_size // 2, z_size]

        adj_list = []
        cur_level = wtree.get_leaves()
        for next_count in self.nodes_sizes[1:]:
            cur_level, _, adj = ward_tree.go_up_to_reduce(cur_level, next_count)
            adj_list.append(adj)
        # adj_list contains adj list from 67615->32768...->128
        # we need to transpose each one and them reverse the list

        self.downsample0 = fgl.FGL(self.channel_sizes[0], self.node_sizes[0], self.channel_sizes[1], self.node_sizes[1], adj_list[0])
        self.downsample1 = fgl.FGL(self.channel_sizes[1], self.node_sizes[1], self.channel_sizes[2], self.node_sizes[2], adj_list[1])
        self.downsample2 = fgl.FGL(self.channel_sizes[2], self.node_sizes[2], self.channel_sizes[3], self.node_sizes[3], adj_list[2])
        self.downsample3 = fgl.FGL(self.channel_sizes[3], self.node_sizes[3], self.channel_sizes[4], self.node_sizes[4], adj_list[3])
        self.downsample4 = fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4])

        self.activation0 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2))
        self.activation1 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2))
        self.activation2 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2))
        self.activation3 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2))
        self.activation4 = nn.Sequential(nn.Dropout(dropout_rate))  # nn.Sequential(nn.LeakyReLU(0.2))

        self.contrast_downsample = nn.Sequential(
            fgl.FGL(self.channel_sizes[3], self.node_sizes[3], self.channel_sizes[4], self.node_sizes[4], adj_list[3]),
            nn.Sequential(nn.Dropout(dropout_rate)),  # nn.Sequential(nn.LeakyReLU(0.2)),
            fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4]),
            nn.Sequential(nn.Dropout(dropout_rate)),  # nn.Sequential(nn.LeakyReLU(0.2)),
        )
        self.task_downsample = nn.Sequential(
            fgl.FGL(self.channel_sizes[4], self.node_sizes[4], self.channel_sizes[5], self.node_sizes[5], adj_list[4]),
            nn.Sequential(nn.Dropout(dropout_rate)),  # nn.Sequential(nn.LeakyReLU(0.2)),
        )
        self.study_downsample = nn.Sequential()
        self.contrast_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['c2i'])),
            nn.Sigmoid(),
        )
        self.task_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['t2i'])),
            nn.Sigmoid(),
        )
        self.study_fc = nn.Sequential(
            nn.Linear(self.node_sizes[-1] * self.channel_sizes[-1], len(meta['s2i'])),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N, constants.masked_nnz
        cur_z = x.unsqueeze(1)  # N, 1, constants.masked_nnz
        downsample = getattr(self, 'downsample{}'.format(0))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(0)):
            cur_z = getattr(self, 'residual{}'.format(0))(cur_z)
        if hasattr(self, 'activation{}'.format(0)):
            cur_z = getattr(self, 'activation{}'.format(0))(cur_z)
        downsample = getattr(self, 'downsample{}'.format(1))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(1)):
            cur_z = getattr(self, 'residual{}'.format(1))(cur_z)
        if hasattr(self, 'activation{}'.format(1)):
            cur_z = getattr(self, 'activation{}'.format(1))(cur_z)

        downsample = getattr(self, 'downsample{}'.format(2))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(2)):
            cur_z = getattr(self, 'residual{}'.format(2))(cur_z)
        if hasattr(self, 'activation{}'.format(2)):
            cur_z = getattr(self, 'activation{}'.format(2))(cur_z)

        c_z = cur_z
        downsample = getattr(self, 'downsample{}'.format(3))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(3)):
            cur_z = getattr(self, 'residual{}'.format(3))(cur_z)
        if hasattr(self, 'activation{}'.format(3)):
            cur_z = getattr(self, 'activation{}'.format(3))(cur_z)
        t_z = cur_z

        downsample = getattr(self, 'downsample{}'.format(4))
        cur_z = downsample(cur_z)
        if hasattr(self, 'residual{}'.format(4)):
            cur_z = getattr(self, 'residual{}'.format(4))(cur_z)
        if hasattr(self, 'activation{}'.format(4)):
            cur_z = getattr(self, 'activation{}'.format(4))(cur_z)
        s_z = cur_z

        s_z = self.study_conv(s_z.view(x.shape[0], -1))
        s = self.study_fc(s_z)


        t_z = self.task_conv(t_z).view(x.shape[0], -1)
        t = self.task_fc(torch.cat([s_z, t_z], dim=1).view(x.shape[0], -1))

        c_z = self.contrast_conv(c_z).view(x.shape[0], -1)
        c = self.contrast_fc(torch.cat([s_z, t_z, c_z], dim=1).view(x.shape[0], -1))

        return s, t, c
