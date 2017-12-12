from collections import namedtuple
import numpy as np
import theano.tensor as tensor

#################### tarjan SCC ########################

TarjanContext = namedtuple('TarjanContext',
                                ['g',           
                                 'S',           
                                 'S_set',       
                                 'index',       
                                 'lowlink',     
                                 'T',           
                                 'ret'])        

def _tarjan_head(ctx, v):
        ctx.index[v] = len(ctx.index)
        ctx.lowlink[v] = ctx.index[v]
        ctx.S.append(v)
        ctx.S_set.add(v)
        it = iter(ctx.g.get(v, ()))
        ctx.T.append((it,False,v,None))

def _tarjan_body(ctx, it, v):
        for w in it:
                if w not in ctx.index:
                        ctx.T.append((it,True,v,w))
                        _tarjan_head(ctx, w)
                        return
                if w in ctx.S_set:
                        ctx.lowlink[v] = min(ctx.lowlink[v], ctx.index[w])
        if ctx.lowlink[v] == ctx.index[v]:
                scc = []
                w = None
                while v != w:
                        w = ctx.S.pop()
                        scc.append(w)
                        ctx.S_set.remove(w)
                ctx.ret.append(scc)

def tarjan(g):
        ctx = TarjanContext(
                g = g,
                S = [],
                S_set = set(),
                index = {},
                lowlink = {},
                T = [],
                ret = [])
        main_iter = iter(g)
        while True:
                try:
                        v = next(main_iter)
                except StopIteration:
                        return ctx.ret
                if v not in ctx.index:
                        _tarjan_head(ctx, v)
                while ctx.T:
                        it, inside, v, w = ctx.T.pop()
                        if inside:
                                ctx.lowlink[v] = min(ctx.lowlink[w],
                                                        ctx.lowlink[v])
                        _tarjan_body(ctx, it, v)

############# End tarjan SCC #####################

def sym(voxel):
    
    shape = tensor.shape(voxel)
    batchsize = shape[0]
    result = tensor.alloc(0.0, batchsize, 2, 3)
    
    for i in [1, 2]:
        left1 = voxel[:, 0:16,i,:, :].reshape((batchsize, -1))
        right1 = voxel[:, 31:15:-1, i,:, :].reshape((batchsize, -1))
        p1 = (left1 * right1).sum(axis=1, keepdims = True)

        left2 = voxel[:, :,i, 0:16,:].reshape((batchsize, -1))
        right2 = voxel[:, :,i, 31:15:-1, :].reshape((batchsize, -1))
        p2 = (left2 * right2).sum(axis=1, keepdims = True)

        left3 = voxel[:, :,i, :, 0:16].reshape((batchsize, -1))
        right3 = voxel[:, :,i, :, 31:15:-1].reshape((batchsize, -1))
        p3 = (left3 * right3).sum(axis=1, keepdims = True)

        num = (left1.sum(axis=1, keepdims=True)+ right1.sum(axis=1, keepdims=True)) // 2  
        result = tensor.set_subtensor(result[:,i-1:,0], p1/num)
        result = tensor.set_subtensor(result[:,i-1:,1], p2/num)
        result = tensor.set_subtensor(result[:,i-1:,2], p3/num)
    
    return result.max(axis = 2).max(axis = 1)
    
def SCC(voxel):
    num = []
    shape = tensor.shape(voxel)
    result = tensor.alloc(0.0, 8, 1)    #Change batchsize
    for b in range(8):                  #Change batchsize
        graph = {}
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    if voxel[b, i, 0, j, k] == 1:
                        graph[(i, j, k)] = []
                        neighbors = [(i+1, j, k), (i-1, j, k), (i, j+1, k), (i, j-1, k), (i, j, k+1), (i, j, k-1)]
                        for x, y, z in neighbors:
                            if ((0 <= x < 32) and (0<=y<32) and (0<=z<32) and (voxel[b, x, 0, y, z] == 1)):
                                graph[(i, j, k)].append((x, y, z))
        result = tensor.set_subtensor(result[b,:], len(s.tarjan(graph)))
    return result

def test():
    from scipy.io import loadmat
    def load_label(modelpath):
        #voxel_fn = get_voxel_file(model_id)
        with open(modelpath, 'rb') as f:
            voxel = loadmat(f)['input']
        return voxel

    voxel = np.array(load_label('../ShapeNet/train_voxels/000005/model.mat'))
    voxel32 = np.zeros((1, 32, 32, 32))
    for i in range(32):
        for j in range(32):
            for k in range(32):
                if np.sum(voxel[0, i*8:(i+1)*8, j*8:(j+1)*8, k*8:(k+1)*8].reshape(512)) > 0:
                    voxel32[0, i, j, k] = 1

    print(sym(voxel32))
    print(SCC(voxel32))


# test()