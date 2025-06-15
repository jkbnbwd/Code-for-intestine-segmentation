import re,glob,os,pickle,shutil
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize_3d,dilation,skeletonize
from skimage import measure
#import itk
import matplotlib.pyplot as plt
from datetime import datetime as dt
import cc3d
import sknw
#from read_endpoint import get_gt_co
#from util_for_chen import get_case_ids
from tqdm import tqdm
#import datetime
global base_path
base_path = '/dataT1/Free/qinan/skele/'
time_mark =  dt.now().strftime('%Y%m%d-%H%M%S')

def skele3D(datapath):
    img_nib = nib.load(datapath)
    img_npy = img_nib.get_fdata()
    img = img_npy
    img = img / np.max(img)
    skel = skeletonize_3d(img / np.max(img))
    skeleton_dilated = dilation(skel)
    save_asnii(skeleton_dilated,img_nib.affine,"Skel.nii.gz")
    print('Skele finish...')



def checkpathexist(path):
    """
    check Dose the directory exist or not?
    if not build it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return 
def save_asnii(img,affine,savedpath):
    """
    Save numpy to nii file
    affine should be import
    """
    img_nii = nib.Nifti1Image(img,affine)
    nib.save(img_nii,savedpath)
def addGTsphere(img_npy,case_id):
    #mask_zeros = np.zeros(shape,dtype = np.int32)
    global gt_cos
    gtco =gt_cos[case_id]
    gt_mask = plotsphere(img_npy.shape,radius=7,position=gtco).astype(np.int32)
    return np.maximum(img_npy,gt_mask*78)
def Measureone(img_npy,message):
    """
    Measure one (segmentation result or skeleton)
    """
    
    img_label = cc3d.connected_components(img_npy, connectivity=26)
    img_label = img_label.astype(float)
    print('Number of connected regions',len(np.unique(img_label)))#Label connected regions of an integer array.
    #size of each connected region
    # print('Voxels in each region:')
    # for label_idx in list(np.unique(img_label)):
    #     print(label_idx,np.sum(img_label== label_idx))
    print(message + ' Finished ','green')
    return img_label
def skel3D(case_id):
    """
    Input path should be set here!
    skelentonize one case of 3D intestine and save it 
    return the Measured Seg
    """
    
    expid = '20220524-114328_AMU_15000'
    #dir = '/dataT1/Free/srchen/ileus/IleusAMU_dataset20210921_fvPattern1/{}/crossvalid_Ileus{}_in1mm_8_classesArgmax.nii.gz'.format(expid,case_id) 
    #Outpath = '/dataT1/Free/srchen/ileus/3Dskel/'
    #dirs = glob.glob(basepath)
    #dir = dirs[0]
    #case_id = re.compile('.*crossvalid_Ileus(.*)_in1mm_8_classes.*').match(dir).group(1)
    #expid = '20220524-114328_AMU_15000'
    global base_path
    Outpath ='./path/'
    
    #copy img_range to the output dirctory
    checkpathexist(Outpath)
    #shutil.copyfile(case_id,'./copypro.nii.gz')

    print("Reading nii of {}".format(case_id))
    if not os.path.exists(Outpath):
        os.makedirs(Outpath)
    img_nib = nib.load(case_id)
    img_npy = img_nib.get_fdata()
    #img_npy[img_npy == 49] = 0 # for 4_paintedConnected...label49 is bone
    #img_npy[img_npy == 42] = 0 # for 6_paintedConnectedSeg...label42 is bone
    #img_npy[img_npy > 0] = 1

    #save_asnii(addGTsphere(img_npy,case_id),img_nib.affine,Outpath+case_id+"mask_RowSeg.nii.gz")
    img_label = cc3d.connected_components(img_npy, connectivity=26)
    original_res = len(np.unique(img_label))
    print("label in this case: {}".format(len(np.unique(img_label))))
    

    #Cut half 
    cut_mask = np.zeros_like(img_npy)
    cut_mask[:,:,:(cut_mask.shape[-1])//3] = 1
    #img_npy = np.logical_and(img_npy,cut_mask).astype(np.int32)
    print("Refine the Seg")
    for idx in tqdm(list(np.unique(img_label))):
        interc = np.logical_and(img_label==idx,cut_mask).astype(np.int32)
        size = np.sum(img_label==idx)
        if np.sum(interc) > size//3 or size < 500:
            img_npy = np.logical_and(img_npy,img_label!=idx).astype(np.int32)
    
    
    img_label = cc3d.connected_components(img_npy, connectivity=26)
    iters = len(np.unique(img_label))//20 + 1
    #iters = 2
    print("Refined label in this case: {},dilation iters: {}".format(len(np.unique(img_label)),iters))
    # dilation
    for _ in range(iters):
       img_npy = dilation(img_npy)

    # img_nii = nib.Nifti1Image(img_npy,img_nib.affine)
    # nib.save(img_nii,Outpath+case_id+"mask_{}.nii.gz".format(time_mark))
    # Save refined segmented intestine
    
    
    #save_asnii(addGTsphere(img_npy,case_id),img_nib.affine,Outpath+case_id+"mask_Seg.nii.gz")
    # Save measured one
    Seg_label = Measureone(img_npy,"Dilated Measure {}'s Segmentation result".format(case_id))
    #save_asnii(addGTsphere(Seg_label,case_id),img_nib.affine,Outpath+case_id+"mask_MeasureSeg.nii.gz")

    if os.path.exists(Outpath+case_id+"mask_Skel.nii.gz"):
        print('Nii file of Skel of {} exists,skip this one'.format(case_id))
        #return Seg_label

    print("Processing skeletonization...")
    img = img_npy
    img = img / np.max(img)
    skel = skeletonize_3d(img / np.max(img))
    skeleton_dilated = dilation(skel)
    #print(np.unique(skeleton_dilated))
    # img_nii = nib.Nifti1Image(skeleton_dilated,img_nib.affine)
    # nib.save(img_nii,Outpath+case_id+"mask_skel_{}.nii.gz".format(time_mark))
    save_asnii(skeleton_dilated,img_nib.affine,Outpath+case_id)
    Skel_label=Measureone(skeleton_dilated,"Measure {}'s Skeleton".format(case_id))
    #save_asnii(addGTsphere(Skel_label,case_id),img_nib.affine,Outpath+case_id+"mask_MeasureSkel.nii.gz")
    print("Skeletonization finished")
    #exit()
    return Seg_label,original_res

def connectedregion():
    """
    find the connected region 
    """
    case_id = 'AMU001'
    path = '/dataT1/Free/srchen/ileus/3Dskel/'
    segpath = path + case_id + 'mask.nii.gz'
    skelpath = path + case_id + 'mask_skel.nii.gz'
    def Measureone(dir,message):
        """
        Measure one (segmentation result or skeleton)
        """
        path = '/dataT1/Free/srchen/ileus/3Dskel/'
        img_nib = nib.load(dir)
        img_npy = img_nib.get_fdata()
        print('Read finished '+ message,'green')
        img_label = measure.label(img_npy,connectivity=2).astype(float)
        print('Number of connected regions',len(np.unique(img_label)))#Label connected regions of an integer array.
        
        #size of each connected region 
        for label_idx in list(np.unique(img_label)):
            print(label_idx,np.sum(img_label== label_idx))
        #print(np.sum(img_label))
        img_nii = nib.Nifti1Image(img_label,img_nib.affine)
        nib.save(img_nii,path+case_id+'mask_'+message+".nii.gz")
        print('Measure finished '+ message,'blue')
    #Measureone(segpath,'MeasureSeg')
    Measureone(skelpath,'MeasureSkel') 
def Testsknw():
    """
    skeleton extraction and path generation for 2D demo
    graph.nodes[id]['pts'] : Numpy(x, n), coordinates of nodes points
    graph.nodes[id]['o']: Numpy(n), centroid of the node
    graph.edge(id1, id2)['pts']: Numpy(x, n), sequence of the edge point
    graph.edge(id1, id2)['weight']: float, length of this edge
    """ 
    from skimage.morphology import skeletonize
    from skimage import data
    import sknw
    
    img = data.horse()
    ske = skeletonize(~img).astype(np.uint16)

    #draw image in subplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original skeleton', fontsize=20)

    ax[1].imshow(img, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Longest path extraction(Search on longest spanning tree)', fontsize=20)


    # build graph from skeleton
    graph = sknw.build_sknw(ske)
    # print(graph.nodes(),graph.edges)
    # draw image
    # plt.imshow(img, cmap='gray')
    
    nodes = graph.nodes()
    
    #generate adjacency matrix table 隣接リスト
    table = [[-1 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for (s,e) in graph.edges():
        table[s][e] = int(graph[s][e]['weight'])
        table[e][s] = int(graph[e][s]['weight'])

    #PRIM algorithm to find maximum spanning tree
    new_table = [[-1 for _ in range(len(nodes))] for _ in range(len(nodes))]
    N = len(table)
    visited = [0]
    while(len(visited) < N):
        maxima = -10 
        a = -1
        b = -1
        for m in visited:
            for n in range(N):
                if n not in visited and table[n][m] != -1:
                    if table[n][m] > maxima:
                        maxima = table[n][m]
                        a = m
                        b = n
        visited.append(b)
        new_table[a][b] = table[a][b]
        new_table[b][a] = table[a][b]
    
    # for i in range(len(table)):
    #     print(i,table[i])
    # print()
    # for i in range(len(new_table)):
    #     print(i,new_table[i])
    table = new_table[:]

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        ax[0].plot(ps[:,1], ps[:,0], 'green')
        if table[s][e] != -1:
            ax[1].plot(ps[:,1], ps[:,0], 'green')
    
    #print(graph.edges())
    
    
    # select outside point 
    from collections import defaultdict
    pointcnt = defaultdict(int)
    for edges in graph.edges():
        for p in edges:
            pointcnt[p] += 1
    sidepoint = [p for p in pointcnt if pointcnt[p] == 1]

    # draw node by o
    for i in nodes:
        ps = np.array([nodes[i]['o']])
        #print(ps[:,1], ps[:,0])
        ax[0].annotate(str(i), xy=(ps[:,1], ps[:,0]), xytext=(-1, 1), textcoords='offset points', color='r',weight='heavy')
        ax[1].annotate(str(i), xy=(ps[:,1], ps[:,0]), xytext=(-1, 1), textcoords='offset points', color='r',weight='heavy')
        ax[0].plot(ps[:,1], ps[:,0], 'b.')
        if i in sidepoint:
            ax[1].plot(ps[:,1], ps[:,0], 'b.')
        
    
    # Find start point
    # x_max,y_max = 0,0
    # max_vaule = -10
    # for x in range(len(nodes)):
    #     for y in range(len(nodes)):
    #         if x in sidepoint and y in sidepoint:
    #             continue
    #         if x in sidepoint or y in sidepoint:
    #             if table[x][y] > max_vaule:
    #                 x_max,y_max = x,y
    #                 max_vaule = table[x][y]
    # startpnt = x_max if x_max in sidepoint else y_max
    
    # draw start node
    # ps = np.array([nodes[startpnt]['o']])
    # plt.plot(ps[:,1], ps[:,0], 'ro')
    # ps = graph[x_max][y_max]['pts']
    # plt.plot(ps[:,1], ps[:,0], 'yellow')

    # find the longest path
    # [distance,previous point]
    startpnt_lgst = sidepoint[0]
    dp_lgst = None
    path_lgst = -1
    for startpnt in sidepoint:
        #print('startpnt',startpnt)
        # startpnt = 15
        # for i in range(len(table)):
        #     print(i,list(map(int,table[i])))
        dp = [[-1,-1] for i in range(len(nodes))]
        stack = [startpnt]
        visited = set([startpnt])
        while stack:
            sstack = stack[:]
            newstack = []
            while sstack:
                #print(sstack)
                pnt = sstack.pop()
                for nextpnt in range(len(table[0])):
                    if nextpnt not in visited and table[pnt][nextpnt] > 0:
                        visited.add(nextpnt)
                        newstack.append(nextpnt)
                        if dp[pnt][0]+table[pnt][nextpnt] > dp[nextpnt][0]:
                            dp[nextpnt][0] = dp[pnt][0] + table[pnt][nextpnt]
                            dp[nextpnt][1] = pnt
            stack = newstack[:]
        dp_npy = np.array(dp)[:,0]
        path_length = max(dp_npy)
        #print("Path finished!\nstartpnt:{}\nlength:{}".format(startpnt,path_length),'blue')
        if path_length >= path_lgst:
            #print(path_length,path_lgst)
            path_lgst = path_length
            startpnt_lgst = startpnt
            dp_lgst = dp[:]
    
    #draw the longest path
    #print(startpnt_lgst,'red')
    # for i in range(len(dp)):
    #     print(i,dp_lgst[i])

    dp_npy = np.array(dp_lgst)[:,0]
    #print(dp_npy)
    idx_max = np.argmax(dp_npy) 
    piv = idx_max
    #print(piv,dp_lgst[piv][1])
    pnt_msg = str(piv)
    while dp_lgst[piv][1] != -1:
        ps = graph[piv][dp_lgst[piv][1]]['pts']
        ax[1].plot(ps[:,1], ps[:,0], 'yellow')
        piv = dp_lgst[piv][1]
        pnt_msg += '->' + str(piv)
    print('The longest path is {}.'.format(pnt_msg),'red')
    
    # Title and show 
    fig.tight_layout()
    #plt.title('Show the path generation')
    plt.show()
def graphToMatrix(graph):
    """
    transfer graph to adjacency matrix table
    """
    nodes = graph.nodes()
    table = [[-1 for _ in range(len(nodes))] for _ in range(len(nodes))]
    for (s,e) in graph.edges():
        table[s][e] = int(graph[s][e]['weight'])
        table[e][s] = int(graph[e][s]['weight'])
    return table
def prim(table):
    """
    PRIM algorithm to find maximum spanning tree
    INPUT: adjacency matrix table 隣接リスト
    
    """
    new_table = [[-1 for _ in range(len(table))] for _ in range(len(table))]
    N = len(table)
    visited = [0]
    while(len(visited) < N):
        maxima = -10 
        a = -1
        b = -1
        for m in visited:
            for n in range(N):
                if n not in visited and table[n][m] != -1:
                    if table[n][m] > maxima:
                        maxima = table[n][m]
                        a = m
                        b = n
        visited.append(b)
        new_table[a][b] = table[a][b]
        new_table[b][a] = table[a][b]
    return new_table
def osidepnt(graph):
    """
    return the point outside the skeleton
    """
    from collections import defaultdict
    pointcnt = defaultdict(int)
    for edges in graph.edges():
        for p in edges:
            pointcnt[p] += 1
    sidepoint = [p for p in pointcnt if pointcnt[p] == 1]
    return sidepoint
def findLgstpath(graph):
    """
    Find longest path in skeleton
    Using Dynamic Programming 動的計画法 to find the longest path
    return dp
    """
    sidepoint = osidepnt(graph)
    nodes = graph.nodes()
    table = graphToMatrix(graph)
    table = prim(table)# use revised prim algorithm to find maximum spanning tree
    #print('Finished prim algorithm','blue')
    startpnt_lgst = sidepoint[0]
    dp_lgst = None
    path_lgst = -1
    for startpnt in sidepoint:
        #print('startpnt',startpnt)
        # startpnt = 15
        # for i in range(len(table)):
        #     print(i,list(map(int,table[i])))
        dp = [[-1,-1] for i in range(len(nodes))]
        stack = [startpnt]
        visited = set([startpnt])
        while stack:
            sstack = stack[:]
            newstack = []
            while sstack:
                #print(sstack)
                pnt = sstack.pop()
                for nextpnt in range(len(table[0])):
                    if nextpnt not in visited and table[pnt][nextpnt] > 0:
                        visited.add(nextpnt)
                        newstack.append(nextpnt)
                        if dp[pnt][0]+table[pnt][nextpnt] > dp[nextpnt][0]:
                            dp[nextpnt][0] = dp[pnt][0] + table[pnt][nextpnt]
                            dp[nextpnt][1] = pnt
            stack = newstack[:]
        dp_npy = np.array(dp)[:,0]
        path_length = max(dp_npy)
        #print("Path finished!\nstartpnt:{}\nlength:{}".format(startpnt,path_length),'blue')
        if path_length >= path_lgst:
            #print(path_length,path_lgst)
            path_lgst = path_length
            startpnt_lgst = startpnt
            dp_lgst = dp[:]
    return dp_lgst
def plotsphere(shape, radius, position):

    """
    Generate an n-dimensional spherical mask.

    return True-False mask
    """
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    #print(position,shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)#(radius,radius,radius)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0
def test3dskel():
    """
    skeleton extraction and path generation for 2D demo
    """
    #https://sabopy.com/py/scikit-image-82/
    import numpy as np
    from skimage.morphology import skeletonize,octahedron,ball
    import matplotlib.pyplot as plt
    import sknw

    struc = np.zeros([15,15,15])
    struc[5:8,5:8,1:14] = 1
    struc[1:14,5:8,11:14] = 1
    
    #skeletonize
    skeleton = skeletonize(struc)
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(131,projection='3d')
    ax1.voxels(struc,ec='w',fc='b')
    ax1.grid()
    ax1.set(xlim=(-5,struc.shape[0]+1),ylim=(-1,struc.shape[0]+1),zlim=(-1,struc.shape[0]+1))
    ax1.set_title('Original', fontsize=15)

    ax2 = fig.add_subplot(132,projection='3d')
    ax2.voxels(skeleton,fc='k',ec='w')
    ax2.grid()
    ax2.set(xlim=(-1,skeleton.shape[0]+1),
        ylim=(-1,skeleton.shape[0]+1),
        zlim=(-1,skeleton.shape[0]+1))
    ax2.set_title('Skeleton', fontsize=15)

    graph = sknw.build_sknw(skeleton)
    print("Node cnt:",len(graph.nodes))
    def plotskel(graph,shape):
        skel_mask = np.zeros(shape)
        for (e,s) in graph.edges():
            line = graph[e][s]['pts']
            skel_mask[line[:,0],line[:,1],line[:,2]] = 1
        return skel_mask
    def plotnode(graph,shape):
        node_mask = np.zeros(shape)
        nodes = graph.nodes()
        coor = np.array([nodes[i]['o'] for i in nodes])
        coor = coor.astype(np.uint16)
        node_mask[coor[:,0],coor[:,1],coor[:,2]] = 1
        return node_mask
    def plotpath(graph,shape):
        """
        plot the longest path,
        return the mask of path and the start point and the end point's coordinary
        """
        print("start finding longest path",'blue')
        dp = findLgstpath(graph)
        path_mask = np.zeros(shape)
        dp_npy = np.array(dp)[:,0]
        idx_max = np.argmax(dp_npy) 
        piv = idx_max
        while dp[piv][1] != -1:
            ps = graph[piv][dp[piv][1]]['pts']
            path_mask[ps[:,0],ps[:,1],ps[:,2]] = 1
            piv = dp[piv][1]
        nodes = graph.nodes()
        return path_mask,[nodes[idx_max]['o'],nodes[piv]['o']]

    node_mask = plotnode(graph,struc.shape)
    path_mask,pnts = plotpath(graph,struc.shape)
    sphere_mask = np.zeros(struc.shape,dtype=bool)
    for pnt in pnts:
        sphere_mask |= plotsphere(shape=struc.shape,radius=1,position=(tuple(pnt.astype(int))))
    ax3 = fig.add_subplot(133,projection='3d')
    ax3.voxels(skeleton,fc='k',ec='w')
    #ax3.voxels(skel_mask,fc='c',ec='w')
    ax3.voxels(path_mask,fc='c',ec='w')
    ax3.voxels(node_mask,fc='y',ec='w')
    ax3.voxels(sphere_mask,fc='m',ec = 'w')
    ax3.set(xlim=(-5,struc.shape[0]+1),ylim=(-1,struc.shape[0]+1),zlim=(-1,struc.shape[0]+1))
    ax3.set_title('Path extraction', fontsize=15)
    fig.tight_layout()
    plt.show()
def plotlinetest():
    """
    Just for test
    plot a line on numpy.array
    """
    #https://sabopy.com/py/scikit-image-82/
    import numpy as np
    from skimage.morphology import skeletonize,octahedron,ball
    import matplotlib.pyplot as plt
    import sknw
    struc = np.zeros([15,15,15])
    struc[7:8,7:8,1:14] = 1
    struc[1:14,7:8,13:14] = 1
    pnt_mask = np.zeros_like(struc)
    #pnt1 = [7,7,1]
    #pnt2 = [13,7,13]
    pnt1 = [1,1,1]
    pnt2 = [10,10,10]
    pnt_mask[tuple(pnt1)] = 1 
    pnt_mask[tuple(pnt2)] = 1
    [dx,dy,dz] = [abs(p1-p2) for p1,p2 in zip(pnt1,pnt2)]
    maxd = max([dx,dy,dz])
    [xx,yy,zz] = [float(d/maxd) for d in [dx,dy,dz]]
    x = [int(min(pnt1[0],pnt2[0])+(i*xx)) for i in range(maxd)]
    y = [int(min(pnt1[1],pnt2[1])+(i*yy)) for i in range(maxd)]
    z = [int(min(pnt1[2],pnt2[2])+(i*zz)) for i in range(maxd)]
    pnt_mask[x,y,z] = 1
    ax = plt.figure().add_subplot(projection = '3d')
    ax.voxels(struc,ec='w',fc='b')
    ax.voxels(pnt_mask,ec='w',fc='r')
    ax.grid()
    #ax.axis('off')
    ax.set(xlim=(-5,struc.shape[0]+1),ylim=(-1,struc.shape[0]+1),zlim=(-1,struc.shape[0]+1))
    ax.set_title('Original', fontsize=15)
    plt.show()
def plotlinenumpy(shape,pnt1,pnt2):
    """
    pnt1 and pnt2 should be the list of coordinate(example: [x,y,z])
    plot in 3D-numpy a line from pnt1 to pnt2
    return the ploted mask
    """
    #print("pnt1:{},pnt2:{}".format(pnt1,pnt2),'green')
    pnt_mask = np.zeros(shape,dtype = np.int32)
    from skimage.draw import line_nd
    lin = line_nd(pnt1,pnt2,endpoint = True)
    pnt_mask[lin] = 1
    return pnt_mask
def recordtime(lasttime,message):
    nowtime = dt.now()
    print(message,(nowtime-lasttime))#.strftime("%H:%M:%S"))
    return nowtime
def extractoneregion(case_id,id):
    """
    select one demo connected region and save it
    return saved path
    """
    print('Extract index:',id)
    #case_id = 'AMU001'
    global base_path
    path = base_path + '{}/path={}/'.format(case_id,id)
    if not os.path.exists(path):# if path does not exist, make it 
        os.makedirs(path)
    # if os.path.exists(path+case_id+'mask_'+'Seg_pathid={}'.format(id)+".nii.gz"):# if it has benn generated
    #     print('Nii file of Path {} exists,skip extract this one'.format(id),'red')
    #     return 
    
    skelpath = base_path + '{}/'.format(case_id) + case_id + 'mask_MeasureSeg.nii.gz'
    img_nib = nib.load(skelpath)
    img_npy = img_nib.get_fdata()
    print('Read connectedpath {} finished!'.format(id),'green')
    img_npy[img_npy != id] = 0
    img_nii = nib.Nifti1Image(img_npy,img_nib.affine)
    outputpath = path+case_id+'mask_'+'Seg_pathid={}'.format(id)+".nii.gz"
    nib.save(img_nii,outputpath)
    print('Path {} of {} saved!'.format(id,case_id),'blue')
    return outputpath
def connectskel(img):
    """
    When the situation that skeleton is not connected
    This function is to connect them
    return the connected mask
    """
    label_len = len(np.unique(img)) - 1
    Outpath = '/dataT1/Free/srchen/ileus/3Dskel/'
    print("There {} connneted regions".format(label_len),'red')
    #print(np.unique(img))
    coods = [] # coordinate of endpoints of each path
    lgstpath_mask = np.zeros_like(img,dtype=np.int32)
    for i in range(1,label_len+1):
        img_i = np.where(img == i, 1, 0)# skeleton of index = i
        
        ### test saving

        global img_nib
        #print(img_nib.affine)
        #print(np.unique(img_i),img_i.dtype)
        img_i = img_i.astype(np.int32)
        #img_dilate = dilation(img_i)
        #img_nii = nib.Nifti1Image(img_dilate,img_nib.affine)
        #nib.save(img_nii,Outpath+"test_{}.nii.gz".format(i))
        
        print("Processing reigon {}... Region size is {}".format(i,np.sum(img_i == 1)))
        if np.sum(img_i == 1) < 20:
            print("Reigon {} is too small to give up it".format(i),'red')
            continue
        
        graph = sknw.build_sknw(img_i)
        dp = findLgstpath(graph)
        
        # extract skel in graph
        skelgraph_mask = np.zeros_like(img_i,dtype=np.int32)
        piv = np.argmax(np.array(dp)[:,0])
        while dp[piv][1] != -1:
            ps = graph[piv][dp[piv][1]]['pts']
            skelgraph_mask[ps[:,0],ps[:,1],ps[:,2]] = 1
            piv = dp[piv][1]
        
        pnts = [np.argmax(np.array(dp)[:,0]),np.argmin(np.array(dp)[:,0])]
        #graph.nodes[id]['o']: Numpy(n), centroid of the node
        pnts_co = [graph.nodes[id]['o'] for id in pnts]
        lgstpath_mask = np.maximum(lgstpath_mask,skelgraph_mask)
        coods.append(pnts_co)
    # np.linalg.norm(vector1-vector2) calculate the Euclidean Distance of two points
    if len(coods) == 1:
        for i in range(1,label_len+1):
            img_i = np.where(img == i, 1, 0)# skeleton of index = i
            print("Processing reigon {}... Region size is {}".format(i,np.sum(img_i == 1)))
            if np.sum(img_i == 1) > 20:
                # print("Reigon {} is too small to give up it".format(i),'red')
                return img_i
    #Connect seperate region together
    def distance(pnts1,pnts2):
        """
        Input: two endpoint's coor of one path
        return: pair of endpoints's coor  and  min_dist of this two path
        """
        min_dist = float('inf')
        res_pair = None
        for p1 in pnts1:
            for p2 in pnts2:
                dist = np.linalg.norm(p1-p2)
                if dist < min_dist:
                    min_dist = dist
                    res_pair = [p1,p2]
        return res_pair,min_dist
    pairs = []
    while(len(coods)>1):
        length = len(coods)
        min_dist = float('inf')
        min_pair = [None,None]
        min_nodes = [None,None]
        for i in range(length):
            for j in range(i+1,length):
                pair_t,dist_t = distance(coods[i],coods[j])
                if dist_t < min_dist:
                    min_dist = dist_t
                    min_pair = pair_t
                    min_nodes = [i,j]
        #print(min_dist,min_pair,min_nodes)
        pairs.append(min_pair)
        tmp_coods = []
        for i in range(length):
            if i not in min_nodes:
                tmp_coods.append(coods[i])
        tmp_coods.append(coods[min_nodes[1]]+coods[min_nodes[0]])
        coods = tmp_coods[:]
        # print(len(coods))
    # print('PAIRS',pairs)
    # print(len(coods),coods)
    # for i in range(len(coods)):
    #     tmp = coods[:]
    #     tmp.pop(i)
    #     tmp = [j for i in tmp for j in i]
    #     pair = None
    #     min_dist = float('inf')
    #     for pnt1 in coods[i]:
    #         for pnt2 in tmp:
    #             dist = np.linalg.norm(pnt1-pnt2)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 pair = [pnt1,pnt2]
    #     flag = False
    #     for p in pairs:
    #         if np.all(p[0] == pair[0]) and np.all(p[1] == pair[1]):
    #             flag = True
    #         if np.all(p[0] == pair[1]) and np.all(p[1]== pair[0]):
    #             flag = True
    #     if not flag:# no same pair in pairs
    #         pairs.append(pair)
    # Draw connections
    cnt_mask = np.zeros_like(img,dtype = np.int32)
    for p in pairs:
        cnt_mask = np.maximum(np.maximum(lgstpath_mask,(plotlinenumpy(cnt_mask.shape,p[0],p[1]).astype(np.int32))),cnt_mask)
    # print(np.sum(cnt_mask))
    cnt_label = cc3d.connected_components(cnt_mask, connectivity=26)
    maxsize = -1
    cnt_maskmax = None
    if len(np.unique(cnt_label)) > 2:
        for i in range(1,len(np.unique(cnt_label))):
            size = np.sum(cnt_label == i)
            if size > maxsize:
                maxsize = size
                cnt_maskmax = np.where(cnt_label == i,1,0)
            print('label {}, Size: {}'.format(i,size),'red')
        cnt_mask = cnt_maskmax
    return cnt_mask
def Process_onecnt(case_id,pid):
    """
    Process one connected region
    1. Skeletonization
    2. Spanning tree generation 
    3. Find the longgest path of each connected skeleton and connect them together
    (The result of spaning tree algorithm may not be connected together)
    4. Check the longgeest path is coorectly generated
    5. Find the endpoints of each path and Draw sphere(Blue(Label:2)) on each endpoint 
    return:
    original skel 
    the processed longgest path mask with sphere
    the processed longgest path mask without sphere
    two endpoints of this cnt path
    """
    #This function is used for test
    def plotskel(graph,shape):
        """
        Draw the skel from graph
        """
        skel_mask = np.zeros(shape)
        for (e,s) in graph.edges():
            line = graph[e][s]['pts']
            skel_mask[line[:,0],line[:,1],line[:,2]] = 1
        return skel_mask
    def saveincolor(img):
        eps = np.where((img ==2),1,0)*2
        skel = np.where((img>0)&(img !=2),pid,0)
        return np.maximum(eps,skel).astype(np.int32)
    #Read from Inpath
    print("Process path {}".format(pid),'blue')
    starttime = dt.now()
    Inpath = '/dataT1/Free/srchen/ileus/3Dskel/{}/path={}/{}mask_Seg_pathid={}.nii.gz'.format(case_id,pid,case_id,pid)
    Outpath = '/dataT1/Free/srchen/ileus/3Dskel/{}/path={}/'.format(case_id,pid)
    global img_nib
    img_nib = nib.load(Inpath)
    img_npy = img_nib.get_fdata()
    readtime = recordtime(starttime,'Reading nii file time')
    #print(np.unique(img_npy))
    
    # Refine the npy data 
    img_npy[img_npy>0] = 1.0
    
    # Extract the skeleton and analyze it 
    skeleton = skeletonize(img_npy)
    skeltime = recordtime(readtime,'Skeletonization time')
    skel_mask_dalated = dilation(skeleton)
    save_asnii(saveincolor(skel_mask_dalated),img_nib.affine,Outpath+"AMU001_skel_pathid={}.nii.gz".format(pid))

    graph = sknw.build_sknw(skeleton)
    graphtime = recordtime(skeltime,'Sknw graph time')
    print("Node in this graph is {}".format(len(graph.nodes())),"red")
    #print('There are {} nodes in the graph'.format(len(graph.nodes())))
    # print('After skeletonization, eps:{}'.format(len(osidepnt(graph))*2),'magenta')
    #sidepoint = osidepnt(graph)
    #print("labelcnt in ske {}".format(len(np.unique(skeleton))),"red")
    
    # Generate spanning tree
    table = graphToMatrix(graph)
    table = prim(table)# use revised prim algorithm to find maximum spanning tree
    spantreetime = recordtime(graphtime,'Span Tree time')
    shape = img_npy.shape
    spantree_mask = np.zeros(shape)
    for e in range(len(table)):
        for s in range(len(table)):
            if table[e][s] != -1:
                line = graph[e][s]['pts']
                spantree_mask[line[:,0],line[:,1],line[:,2]] = 1
    #save spantree
    spantree_mask_dalated = dilation(spantree_mask)
    save_asnii(saveincolor( spantree_mask_dalated),img_nib.affine,Outpath+"AMU001_spaentree_pathid={}.nii.gz".format(pid))
    savespantime = recordtime(spantreetime,'Saved spantree nii file') 

    #Process some situation that some unconnected region appear after skeleton extraction
    #skeleton = plotskel(graph,shape)
    img_label = cc3d.connected_components(spantree_mask, connectivity=26)
    #print('skel regions',len(np.unique(img_label)))
    if len(np.unique(img_label)) > 2:
        print("Skel need to be connected","red")
        cnt_mask = connectskel(img_label) #mask after connect together
    elif len(np.unique(img_label)) == 1:
        print("There no voxel after skeletonization, skip the cnt {}".format(pid),"red")
        npy_zeros = np.zeros_like(img_label,dtype=np.int32)
        return npy_zeros,npy_zeros,npy_zeros,[]
    else:#all conponent already connected together
        cnt_mask = spantree_mask
    
    # save cnt
    cntmask_dalated = dilation(cnt_mask)
    save_asnii(saveincolor( cntmask_dalated),img_nib.affine,Outpath+"AMU001_cntmask_pathid={}.nii.gz".format(pid))
    savecnt = recordtime(savespantime,'Saved connected skel nii file')

    #Check again and find the longgest path and draw endnode on this one 
    img_label = cc3d.connected_components(cnt_mask, connectivity=26)
    if len(np.unique(img_label)) == 2:
        print('All conponent connected together!','green')
        skeleton_cnt = skeletonize(cnt_mask.astype(np.uint8))
        graph = sknw.build_sknw(skeleton_cnt)
        # print('After generate spantree, eps:{}'.format(len(osidepnt(graph))),'magenta')
        fin_dp = findLgstpath(graph)
        #graph.nodes[id]['o'] for id in pnts
        fin_mask = np.zeros_like(cnt_mask,dtype = np.int32)
        piv = np.argmax(np.array(fin_dp)[:,0])
        while fin_dp[piv][1] != -1:
            ps = graph[piv][fin_dp[piv][1]]['pts']
            fin_mask[ps[:,0],ps[:,1],ps[:,2]] = 1
            piv = fin_dp[piv][1]
        endpoints = []
        for id in [np.argmax(np.array(fin_dp)[:,0]),np.argmin(np.array(fin_dp)[:,0])]:
            pnt = graph.nodes[id]['o']
            endpoints.append(pnt)
            #print("Endpoint of path{}".format(pid),pnt)
            sphere_mask = ((plotsphere(shape=cnt_mask.shape,radius=2,position=(tuple(pnt.astype(int))))).astype(np.int32))*102
            fin_mask = np.maximum(sphere_mask,fin_mask)
            #save candidate_mask
        # print("Endpoint of path{}: {}".format(pid,endpoints))
        # if len(graph.nodes()) == 2:
        #     for id in range(len(graph.nodes())):
        #         pnt = graph.nodes[id]['o']
        #         print(pnt)
        #         sphere_mask = ((plotsphere(shape=cnt_mask.shape,radius=2,position=(tuple(pnt.astype(int))))).astype(np.int32))*2
        #         fin_mask = np.maximum(np.maximum(cnt_mask,sphere_mask),fin_mask)
    else:
        print('Still has {} unconnected regions!'.format(len(np.unique(img_label))-1),'red')
        print("Sum: ",np.sum(cnt_mask))
        for i in range(1,len(np.unique(img_label))):
            size = np.sum(img_label == i)
            print('label {}, Size: {}'.format(i,size),'red')
        #exit()
        fin_mask = cnt_mask
        endpoints = []
    #save final mask 
    
    finmask_dalated = dilation(fin_mask)
    save_asnii(saveincolor(finmask_dalated),img_nib.affine,Outpath+"AMU001_finalmask_pathid={}.nii.gz".format(pid))
    savetime = recordtime(savecnt,'Saved final nii file')
    sumtime = recordtime(starttime,'Processd path {} time'.format(pid))
    return skeleton,finmask_dalated,cntmask_dalated,endpoints
def saveaspickle(data,case_id):
    """
    data should be dict():
    (
        For one case it should include:
        Endpoint coordinate
        Extracted path mask (numpy.arrays)
    )
    save data as pickle file 
    """
    global base_path
    path = os.path.join(base_path,"Pkl_file"+time_mark,case_id+".pickle")
    checkpathexist(os.path.join(base_path,"Pkl_file"+time_mark))
    with open(path,'wb') as f:
        pickle.dump(data,f)
def readpickle(case_id):
    """
    read endpoint coordinate and extracted path mask from pkl
    """
    global base_path
    path = os.path.join(base_path,"Pkl_file",case_id+".pickle")
    if not os.path.exists(path):
        print("The pkl file does not exist","red")
        exit()
    with open(path,"rb") as f:
        data = pickle.load(path)
    return data
def Process_onecase(case_id):
    """
    Process 3D skel for one case
    """
    case_starttime = dt.now().replace(microsecond=0)
    global gt_cos
    print("Processing {} start! Intestinal obstruction GT CO is {}".format(case_id,gt_cos[case_id]))
    starttime = dt.now()
    Seg_label,org_cnt = skel3D(case_id)#origianl_cnt is original cnt region 
    #measureskeltime= recordtime(starttime,"Finished save measure and skel nii file time")
    print("Measure Seg and Skel finished!")
    pathids = list(np.unique(Seg_label.astype(np.int)))
    print('Cnt count',len(pathids))
    
    # Read the Measured Seg
    global base_path
    Segpath = base_path + '{}/'.format(case_id) + case_id + 'mask_MeasureSeg.nii.gz'
    img_nib = nib.load(Segpath)
    #img_npy = img_nib.get_fdata()

    sum_skelmask = np.zeros_like(Seg_label,dtype=np.int32)
    sum_finmask = np.zeros_like(Seg_label,dtype=np.int32)
    sum_nospheremask = np.zeros_like(Seg_label,dtype=np.int32)
    sum_eps = []
    
    #minpsize
    minpsize = 2300

    print("Start Processing cnt")
    for pid in tqdm(pathids[1:]):
        startpidtime = dt.now()
        psize = np.sum(Seg_label == pid)
        if psize < minpsize:# if Seg path size < minpize: Skip this one
            continue
        print("Process {} connected region(regrad as a path),size of it is {}".format(pid,psize),"blue")
        
        extractoneregion(case_id,pid)
        print("Finished extract region")
        pid_skelmask, pid_finmask,pid_nospheremask,eps = Process_onecnt(case_id,pid)
        
        pid_skelmask[pid_skelmask > 0] = pid
        pid_finmask[pid_finmask == 1] = pid
        pid_nospheremask[pid_nospheremask == 1] = pid
        
        sum_skelmask += pid_skelmask
        sum_finmask += pid_finmask
        sum_nospheremask += pid_nospheremask.astype(np.int32)
        sum_eps += eps

        endpidtime = recordtime(startpidtime,"Finished process onecnt")
        print("Finished Process this region")
    endtime = recordtime(starttime,"Finished the case")
    
    #addGTsphere(,case_id)
    # Save mask
    #save_asnii(addGTsphere(dilation(sum_skelmask),case_id),img_nib.affine,base_path+"/{}/{}_sumskelmask.nii.gz".format(case_id,case_id))
    #save_asnii(addGTsphere(sum_finmask,case_id),img_nib.affine,base_path+"/{}/{}_sumfinmask.nii.gz".format(case_id,case_id))
    #save_asnii(addGTsphere(sum_nospheremask,case_id),img_nib.affine,base_path+"/{}/{}_sumnospheremask.nii.gz".format(case_id,case_id))
    
    # Save
    #print("Before processing,there are {} endpoints in case {}".format((len(np.unique(Seg_label))-1)*2,case_id))
    print("Before processing,there are {} endpoints in case {}".format(org_cnt*2,case_id))
    print("After processing,there are {} endpoints in case {}".format(len(sum_eps),case_id))
    #global gt_cos
    gt_co = gt_cos[case_id]
    dist_gt = [np.linalg.norm(gt_co-ep) for ep in sum_eps]
    print(len(dist_gt),dist_gt)
    if len(dist_gt) == 0:
        print("There is no endpoint in {}".format(case_id),"red")
        return
    print("Min-dist = {}".format(min(dist_gt)))
    
    case_endtime = dt.now().replace(microsecond=0)
    case_time = str(case_endtime-case_starttime)
    print("Case Time:",case_time)
    datapkl = {
        'mindist':min(dist_gt),
        'dist_gt':dist_gt,
        'eps':sum_eps,
        'mask':sum_nospheremask,
        'casetime':case_time,
        'minpsize':minpsize,
        'flag':'no_a'
    }
    saveaspickle(datapkl,case_id)


    print("Saved as pickle file. Finished the case {}".format(case_id),"blue")

def proconebyone():
    """
    Process the case one by one
    It is easy to debug
    """
    case_ids = get_case_ids()
    global gt_cos
    gt_cos = get_gt_co()
    for case_id in case_ids[:1]:
        Process_onecase(case_id)
def Multijob():
    """
    Process the case using multijob
    It is faster
    """
    from multiprocessing import cpu_count
    from psutil import virtual_memory
    from joblib import Parallel,delayed
    mem = virtual_memory()
    availGB = mem.available / 1024 / 1024 / 1024
    regardingMem = int(availGB / 4)
    PARALLEL =  min(min(regardingMem, cpu_count()-1), 79) - 1
    
    global gt_cos
    gt_cos = get_gt_co()
    caseids = get_case_ids()
    Parallel(n_jobs = PARALLEL)(delayed(Process_onecase)(id) for id in caseids)

    
if __name__ == '__main__':
    #
    #connectedregion()
    #Testsknw()
    #test3dskel()
    #demo_one()
    #plotlinetest()
    #global gt_cos
    #gt_cos = get_gt_co()
    case_id = 'labels-IleusAMU125_in1mm_phase0.nii.gz'
    #Process_onecase(case_id)
    skel3D(case_id)
    #skele3D(case_id)
    # Multijob()
    # from eval_postprocessing import read_saveResult
    # #proconebyone()
    # read_saveResult()
    



        
