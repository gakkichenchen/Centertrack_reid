import sys,os,copy,math
from munkres import Munkres
from collections import defaultdict
try:
    from ordereddict import OrderedDict # can be installed using pip
except:
    from collections import OrderedDict # only included from python 2.7 on


class tData:
    def __init__(self,frame=-1,obj_type="unset",truncation=-1,occlusion=-1,\
                 obs_angle=-10,x1=-1,y1=-1,x2=-1,y2=-1,w=-1,h=-1,l=-1,\
                 X=-1000,Y=-1000,Z=-1000,yaw=-10,score=-1000,track_id=-1):
        # init object data
        self.frame      = frame
        self.track_id   = track_id
        self.obj_type   = obj_type
        self.truncation = truncation
        self.occlusion  = occlusion
        self.obs_angle  = obs_angle
        self.x1         = x1
        self.y1         = y1
        self.x2         = x2
        self.y2         = y2
        self.w          = w
        self.h          = h
        self.l          = l
        self.X          = X
        self.Y          = Y
        self.Z          = Z
        self.yaw        = yaw
        self.score      = score
        self.ignored    = False
        self.valid      = False
        self.tracker    = -1

    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

class trackingEvaluation(object):
    def __init__(self,cls,t_path='/home/ubuntu/cc/scripts/sort-master/outputjson/result3', gt_path="/home/ubuntu/cc/scripts/sort-master/outputjson/test_txt",\
        split_version='', min_overlap=0.5, max_truncation = 0, min_height = 25, 
        max_occlusion = 2):
        self.t_path=t_path#tracker
        self.gt_path=gt_path#groundtrue
        filename_test_mapping = "/home/ubuntu/cc/scripts/sort-master/outputjson/eval_holo_track/holo_data_test.txt"
        self.n_frames         = []
        self.sequence_name    = []
        with open(filename_test_mapping, "r") as fh:
            for i,l in enumerate(fh):
                fields = l.split(" ")
                self.sequence_name.append(fields[0])#the video dir
                self.n_frames.append(int(fields[1])+1)#the frame num
        fh.close()
        self.n_sequences = i+1#the video num
        self.cls = cls
        
        # statistics and numbers for evaluation
        self.n_gt              = 0 # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt             = 0 # number of ignored ground truth detections
        self.n_gts             = [] # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts            = [] # number of ground ignored truth detections PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_seq          = []
        self.n_tr              = 0 # number of tracker detections minus ignored tracker detections
        self.n_trs             = [] # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itr             = 0 # number of ignored tracker detections
        self.n_itrs            = [] # number of ignored tracker detections PER SEQUENCE
        self.n_igttr           = 0 # number of ignored ground truth detections where the corresponding associated tracker detection is also ignored
        self.n_tr_trajectories = 0
        self.n_tr_seq          = []
        self.MOTA              = 0
        self.MOTP              = 0
        self.MOTAL             = 0
        self.MODA              = 0
        self.MODP              = 0
        self.MODP_t            = []
        self.recall            = 0
        self.precision         = 0
        self.F1                = 0
        self.FAR               = 0
        self.total_cost        = 0
        self.itp               = 0 # number of ignored true positives
        self.itps              = [] # number of ignored true positives PER SEQUENCE
        self.tp                = 0 # number of true positives including ignored true positives!
        self.tps               = [] # number of true positives including ignored true positives PER SEQUENCE
        self.fn                = 0 # number of false negatives WITHOUT ignored false negatives
        self.fns               = [] # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifn               = 0 # number of ignored false negatives
        self.ifns              = [] # number of ignored false negatives PER SEQUENCE
        self.fp                = 0 # number of false positives
                                   # a bit tricky, the number of ignored false negatives and ignored true positives 
                                   # is subtracted, but if both tracker detection and ground truth detection
                                   # are ignored this number is added again to avoid double counting
        self.fps               = [] # above PER SEQUENCE
        self.mme               = 0
        self.fragments         = 0
        self.id_switches       = 0
        self.MT                = 0
        self.PT                = 0
        self.ML                = 0
        
        self.min_overlap       = min_overlap # minimum bounding box overlap for 3rd party metrics 0.5
        self.max_truncation    = max_truncation # maximum truncation of an object for evaluation 0
        self.max_occlusion     = max_occlusion # maximum occlusion of an object for evaluation 2
        self.min_height        = min_height # minimum height of an object for evaluation 25
        self.n_sample_points   = 500
        
        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories            = [[] for x in range(self.n_sequences)]#26
        self.ign_trajectories           = [[] for x in range(self.n_sequences)]#26


    def loadGroundtruth(self):
        """
            Helper function to load ground truth.
        """
        
        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadTracker(self):
        try:
            if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True
    #gt:groundtrue   t:tracker
    def _loadData(self, root_dir, cls, min_score=-1000, loading_groundtruth=False):
        t_data  = tData()
        data    = []
        eval_2d = True
        eval_3d = True

        seq_data           = []
        n_trajectories     = 0
        n_trajectories_seq = []
        #video dir
        for seq, s_name in enumerate(self.sequence_name):
            i              = 0
            filename       = os.path.join(root_dir, "%s.txt" % s_name)
            f              = open(filename, "r")

            f_data         = [[] for x in range(self.n_frames[seq])]#[frame1,frame2,frame3]
             # current set has only 1059 entries, sufficient length is checked anyway
            ids            = []
            n_in_seq       = 0
            id_frame_cache = []
            for line in f:
                line = line.strip()
                fields            = line.split(" ")
                # classes that should be loaded (ignored neighboring classes)
                if "car" in cls.lower():
                    classes = ["car"]
                elif "people" in cls.lower():
                    classes = ['people']
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # get fields from table
                t_data.frame        = int(float(fields[0]))     # frame
                t_data.track_id     = int(float(fields[1]))     # id
                t_data.obj_type     = fields[2].lower()         # object type [car, pedestrian, cyclist, ...]
                t_data.truncation   = int(float(fields[3]))     # truncation [-1,0,1,2]
                t_data.occlusion    = int(float(fields[4]))     # occlusion  [-1,0,1,2]
                t_data.obs_angle    = float(fields[5])          # observation angle [rad]
                t_data.x1           = float(fields[6])          # left   [px]
                t_data.y1           = float(fields[7])          # top    [px]
                t_data.x2           = float(fields[8])          # right  [px]
                t_data.y2           = float(fields[9])          # bottom [px]
                t_data.h            = float(fields[10])         # height [m]
                t_data.w            = float(fields[11])         # width  [m]
                t_data.l            = float(fields[12])         # length [m]
                t_data.X            = float(fields[13])         # X [m]
                t_data.Y            = float(fields[14])         # Y [m]
                t_data.Z            = float(fields[15])         # Z [m]
                t_data.yaw          = float(fields[16])         # yaw angle [rad]
                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score  = float(fields[17])     # detection score
                    else:

                        return

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "others":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    print("extend f_data", idx, len(f_data))
                    f_data += [[] for x in range(max(500, idx-len(f_data)))]
                try:
                    id_frame = (t_data.frame,t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        #continue # this allows to evaluate non-unique result files
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print(len(f_data), idx)
                    raise

                if t_data.track_id not in ids and t_data.obj_type!="others":
                    ids.append(t_data.track_id)
                    n_trajectories +=1
                    n_in_seq +=1

                # check if uploaded data provides information for 2D and 3D evaluation
                if not loading_groundtruth and eval_2d is True and(t_data.x1==-1 or t_data.x2==-1 or t_data.y1==-1 or t_data.y2==-1):
                    eval_2d = False
                if not loading_groundtruth and eval_3d is True and(t_data.X==-1000 or t_data.Y==-1000 or t_data.Z==-1000):
                    eval_3d = False

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()
        if not loading_groundtruth:
            self.tracker=seq_data
            self.n_tr_trajectories=n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories==0:
                return False
        else:
            # split ground truth and DontCare areas
            self.dcareas     = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [],[]
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g,dc = [],[]
                    for gg in all_gt:
                        if gg.obj_type=="others":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq=n_trajectories_seq
            self.n_gt_trajectories=n_trajectories#the num of objects
        return True


    def boxoverlap(self,a,b,criterion="union"):
        """
            boxoverlap computes intersection over union for bbox a and b in KITTI format.
            If the criterion is 'union', overlap = (a inter b) / a union b).
            If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        """
        
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        w = x2-x1
        h = y2-y1

        if w<=0. or h<=0.:
            return 0.
        inter = w*h
        aarea = (a.x2-a.x1) * (a.y2-a.y1)
        barea = (b.x2-b.x1) * (b.y2-b.y1)
        # intersection over union overlap
        if criterion.lower()=="union":
            o = inter / float(aarea+barea-inter)
        elif criterion.lower()=="a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("Unkown type for criterion")
        return o

    def compute3rdPartyMetrics(self):
        """
            Computes the metrics defined in
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0,0
        for seq_idx in range(len(self.groundtruth)):
            seq_gt                = self.groundtruth[seq_idx]
            seq_dc                = self.dcareas[seq_idx] # don't care areas
            seq_tracker           = self.tracker[seq_idx]
            seq_trajectories      = defaultdict(list)
            seq_ignored           = defaultdict(list)
            
            # statistics over the current sequence, check the corresponding
            # variable comments in __init__ to get their meaning
            seqtp            = 0
            seqitp           = 0
            seqfn            = 0
            seqifn           = 0
            seqfp            = 0
            seqigt           = 0
            seqitr           = 0
            
            last_ids = [[],[]]
            
            n_gts = 0
            n_trs = 0
            
            for f in range(len(seq_gt)):
                g = seq_gt[f]
                dc = seq_dc[f]

                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)
                
                n_gts += len(g)
                n_trs += len(t)
                
                # use hungarian method to associate, using boxoverlap 0..1 as cost
                # build cost matrix
                cost_matrix = []
                this_ids = [[],[]]
                for gg in g:
                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker       = -1
                    gg.id_switch     = 0
                    gg.fragmentation = 0
                    cost_row         = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1-self.boxoverlap(gg,tt)

                        # gating for boxoverlap
                        if c<=self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost) # = 1e9
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)
                    
                if len(g) is 0:
                    cost_matrix=[[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # tmp variables for sanity checks and MODP computation
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc  = 0 # this will sum up the overlaps for all true positives
                tmpcs = [0]*len(g) # this will save the overlaps for all true positives
                                   # the reason is that some true positives might be ignored
                                   # later such that the corrsponding overlaps can
                                   # be subtracted from tmpc for MODP computation
                
                # mapping for tracker ids and ground truth ids
                for row,col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        g[row].tracker   = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid     = True
                        g[row].distance  = c
                        self.total_cost += 1-c
                        tmpc            += 1-c
                        tmpcs[row]      = 1-c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp   += 1
                    else:
                        g[row].tracker = -1
                        self.fn       += 1
                        tmpfn         += 1
                
                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0 # number of ignored tracker detections
                ignoredtrackers = dict() # will associate the track_id with -1
                                         # if it is not ignored and 1 if it is
                                         # ignored;
                                         # this is used to avoid double counting ignored
                                         # cases, see the next loop
                
                for tt in t:
                    ignoredtrackers[tt.track_id] = -1
                    # ignore detection if it belongs to a neighboring class or is
                    # smaller or equal to the minimum height
                    
                    tt_height = abs(tt.y1 - tt.y2)
                    if ((self.cls=="car" and tt.obj_type=="people") or (self.cls=="people" and tt.obj_type=="car") or tt_height<=self.min_height) and not tt.valid:
                        nignoredtracker+= 1
                        tt.ignored      = True
                        ignoredtrackers[tt.track_id] = 1
                        continue
                    for d in dc:
                        overlap = self.boxoverlap(tt,d,"a")
                        if overlap>0.5 and not tt.valid:
                            tt.ignored      = True
                            nignoredtracker+= 1
                            ignoredtrackers[tt.track_id] = 1
                            break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn  = 0 # the number of ignored false negatives
                nignoredtp = 0 # the number of ignored true positives
                nignoredpairs = 0 # the number of ignored pairs, i.e. a true positive
                                  # which is ignored but where the associated tracker
                                  # detection has already been ignored
                                  
                gi = 0
                for gg in g:
                    if gg.tracker < 0:
                        if gg.occlusion>self.max_occlusion or gg.truncation>self.max_truncation\
                                or (self.cls=="car" and gg.obj_type=="people") or (self.cls=="people" and gg.obj_type=="car"):
                            seq_ignored[gg.track_id][-1] = True                              
                            gg.ignored = True
                            ignoredfn += 1
                            
                    elif gg.tracker>=0:
                        if gg.occlusion>self.max_occlusion or gg.truncation>self.max_truncation\
                                or (self.cls=="car" and gg.obj_type=="people") or (self.cls=="people" and gg.obj_type=="car"):
                            
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1
                            
                            # if the associated tracker detection is already ignored,
                            # we want to avoid double counting ignored detections
                            if ignoredtrackers[gg.tracker] > 0:
                                nignoredpairs += 1
                            
                            # for computing MODP, the overlaps from ignored detections
                            # are subtracted
                            tmpc -= tmpcs[gi]
                    gi += 1
                
                # the below might be confusion, check the comments in __init__
                # to see what the individual statistics represent
                
                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp
                
                # count the number of ignored true positives
                self.itp += nignoredtp
                
                # adjust the number of ground truth objects considered
                self.n_gt -= (ignoredfn + nignoredtp)
                
                # count the number of ignored ground truth objects
                self.n_igt += ignoredfn + nignoredtp
                
                # count the number of ignored tracker objects
                self.n_itr += nignoredtracker
                
                # count the number of ignored pairs, i.e. associated tracker and
                # ground truth objects that are both ignored
                self.n_igttr += nignoredpairs
                
                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                # 
                tmpfn   += len(g)-len(association_matrix)-ignoredfn
                self.fn += len(g)-len(association_matrix)-ignoredfn
                self.ifn += ignoredfn
                
                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t)
                tmpfp   += len(t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                self.fp += len(t) - tmptp - nignoredtracker - nignoredtp + nignoredpairs
                #tmpfp   = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
                #self.fp += len(t) - tmptp - nignoredtp

                # update sequence data
                seqtp += tmptp
                seqitp += nignoredtp
                seqfp += tmpfp
                seqfn += tmpfn
                seqifn += ignoredfn
                seqigt += ignoredfn + nignoredtp
                seqitr += nignoredtracker
                
                # sanity checks
                # - the number of true positives minues ignored true positives
                #   should be greater or equal to 0
                # - the number of false negatives should be greater or equal to 0
                # - the number of false positives needs to be greater or equal to 0
                #   otherwise ignored detections might be counted double
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false negatives (plus ignored ones)
                #   should match the total number of ground truth objects
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false positives
                #   plus the number of ignored tracker detections should
                #   match the total number of tracker detections; note that
                #   nignoredpairs is subtracted here to avoid double counting
                #   of ignored detection sin nignoredtp and nignoredtracker
                if tmptp<0:
                    print(tmptp, nignoredtp)
                    raise NameError("Something went wrong! TP is negative")
                if tmpfn<0:
                    print(tmpfn, len(g), len(association_matrix), ignoredfn, nignoredpairs)
                    raise NameError("Something went wrong! FN is negative")
                if tmpfp<0:
                    print(tmpfp, len(t), tmptp, nignoredtracker, nignoredtp, nignoredpairs)
                    raise NameError("Something went wrong! FP is negative")
                if tmptp + tmpfn is not len(g)-ignoredfn-nignoredtp:
                    print("seqidx", seq_idx)
                    print("frame ", f)
                    print("TP    ", tmptp)
                    print("FN    ", tmpfn)
                    print("FP    ", tmpfp)
                    print("nGT   ", len(g))
                    print("nAss  ", len(association_matrix))
                    print("ign GT", ignoredfn)
                    print("ign TP", nignoredtp)
                    raise NameError("Something went wrong! nGroundtruth is not TP+FN")
                if tmptp+tmpfp+nignoredtp+nignoredtracker-nignoredpairs is not len(t):
                    print(seq_idx, f, len(t), tmptp, tmpfp)
                    print(len(association_matrix), association_matrix)
                    raise NameError("Something went wrong! nTracker is not TP+FP")

                # check for id switches or fragmentations
                for i,tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation<self.max_truncation:
                                g[i].id_switch = 1
                                ids +=1
                        if tid != lid and lid != -1:
                            if g[i].truncation<self.max_truncation:
                                g[i].fragmentation = 1
                                fr +=1

                # save current index
                last_ids = this_ids
                # compute MOTP_t
                MODP_t = 1
                if tmptp!=0:
                    MODP_t = tmpc/float(tmptp)
                self.MODP_t.append(MODP_t)

            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx]             = seq_trajectories
            self.ign_trajectories[seq_idx]            = seq_ignored
            
            # gather statistics for "per sequence" statistics.
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)
        
        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (seq_trajectories,seq_ignored) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories)==0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0]*5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(), seq_ignored.values()):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr+=1
                    n_ignored_tr_total+=1
                    continue
                # all frames of this gt trajectory are not assigned to any detections
                if all([this==-1 for this in g]):
                    tmpML+=1
                    self.ML+=1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0]>=0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1,len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt+=1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f-1] != -1:
                        tmpId_switches   += 1
                        self.id_switches += 1
                    if f < len(g)-1 and g[f-1] != g[f] and last_id != -1 and g[f] != -1 and g[f+1] != -1:
                        tmpFragments   += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handled in for loop (g[f]!=-1)
                if len(g)>1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1 and not ign_g[f]:
                    tmpFragments   += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    tmpMT   += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML   += 1
                    self.ML += 1
                else: # 0.2 <= tracking_ratio <= 0.8
                    tmpPT   += 1
                    self.PT += 1

        if (self.n_gt_trajectories-n_ignored_tr_total)==0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories-n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp+self.tp)==0 or (self.tp+self.fn)==0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp/float(self.tp+self.fn)
            self.precision = self.tp/float(self.fp+self.tp)
        if (self.recall+self.precision)==0:
            self.F1 = 0.
        else:
            self.F1 = 2.*(self.precision*self.recall)/(self.precision+self.recall)
        if sum(self.n_frames)==0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp/float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt==0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA  = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            self.MODA  = 1 - ( self.fn+ self.fp) / float(self.n_gt)
        if self.tp==0:
            self.MOTP  = float("inf")
        else:
            self.MOTP  = self.total_cost / float(self.tp)
        if self.n_gt!=0:
            if self.id_switches==0:
                self.MOTAL = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            else:
                self.MOTAL = 1 - (self.fn + self.fp + math.log10(self.id_switches))/float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames)==0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t)/float(sum(self.n_frames))
        return True

    def createSummary(self):
       
        
        summary = ""
        
        summary += "tracking evaluation summary".center(80,"=") + "\n"
        summary += self.printEntry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA) + "\n"
        summary += self.printEntry("Multiple Object Tracking Precision (MOTP)", self.MOTP) + "\n"
        summary += self.printEntry("Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL) + "\n"
        summary += self.printEntry("Multiple Object Detection Accuracy (MODA)", self.MODA) + "\n"
        summary += self.printEntry("Multiple Object Detection Precision (MODP)", self.MODP) + "\n"
        summary += "\n"
        summary += self.printEntry("Recall", self.recall) + "\n"
        summary += self.printEntry("Precision", self.precision) + "\n"
        summary += self.printEntry("F1", self.F1) + "\n"
        summary += self.printEntry("False Alarm Rate", self.FAR) + "\n"
        summary += "\n"
        summary += self.printEntry("Mostly Tracked", self.MT) + "\n"
        summary += self.printEntry("Partly Tracked", self.PT) + "\n"
        summary += self.printEntry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.printEntry("True Positives", self.tp) + "\n"
        #summary += self.printEntry("True Positives per Sequence", self.tps) + "\n"
        summary += self.printEntry("Ignored True Positives", self.itp) + "\n"
        #summary += self.printEntry("Ignored True Positives per Sequence", self.itps) + "\n"

        summary += self.printEntry("False Positives", self.fp) + "\n"
        #summary += self.printEntry("False Positives per Sequence", self.fps) + "\n"
        summary += self.printEntry("False Negatives", self.fn) + "\n"
        #summary += self.printEntry("False Negatives per Sequence", self.fns) + "\n"
        summary += self.printEntry("ID-switches", self.id_switches) + "\n"
        self.fp = self.fp / self.n_gt
        self.fn = self.fn / self.n_gt
        self.id_switches = self.id_switches / self.n_gt
        summary += self.printEntry("False Positives Ratio", self.fp) + "\n"
        #summary += self.printEntry("False Positives per Sequence", self.fps) + "\n"
        summary += self.printEntry("False Negatives Ratio", self.fn) + "\n"
        #summary += self.printEntry("False Negatives per Sequence", self.fns) + "\n"
        summary += self.printEntry("Ignored False Negatives Ratio", self.ifn) + "\n"

        #summary += self.printEntry("Ignored False Negatives per Sequence", self.ifns) + "\n"
        summary += self.printEntry("Missed Targets", self.fn) + "\n"
        summary += self.printEntry("ID-switches", self.id_switches) + "\n"
        summary += self.printEntry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += self.printEntry("Ground Truth Objects (Total)", self.n_gt + self.n_igt) + "\n"
        #summary += self.printEntry("Ground Truth Objects (Total) per Sequence", self.n_gts) + "\n"
        summary += self.printEntry("Ignored Ground Truth Objects", self.n_igt) + "\n"
        #summary += self.printEntry("Ignored Ground Truth Objects per Sequence", self.n_igts) + "\n"
        summary += self.printEntry("Ground Truth Trajectories", self.n_gt_trajectories) + "\n"
        summary += "\n"
        summary += self.printEntry("Tracker Objects (Total)", self.n_tr) + "\n"
        #summary += self.printEntry("Tracker Objects (Total) per Sequence", self.n_trs) + "\n"
        summary += self.printEntry("Ignored Tracker Objects", self.n_itr) + "\n"
        #summary += self.printEntry("Ignored Tracker Objects per Sequence", self.n_itrs) + "\n"
        summary += self.printEntry("Tracker Trajectories", self.n_tr_trajectories) + "\n"
        #summary += "\n"
        #summary += self.printEntry("Ignored Tracker Objects with Associated Ignored Ground Truth Objects", self.n_igttr) + "\n"
        summary += "="*80
        
        return summary

    def printEntry(self, key, val,width=(70,10)):
        """
            Pretty print an entry in a table fashion.
        """
        
        s_out =  key.ljust(width[0])
        if type(val)==int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val)==float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s"%val).rjust(width[1])
        return s_out
      
    def saveToStats(self):
        """
            Save the statistics in a whitespace separate file.
        """
        
        # create pretty summary
        self.t_sha='/home/ubuntu/cc/scripts/sort-master/outputjson/result3/eval_sh'
        isexit=os.path.exists(self.t_sha)
        if not isexit:
            os.mkdir(self.t_sha)
        summary = self.createSummary()
        filename = os.path.join(self.t_sha, "summary_%s.txt" % self.cls)
        dump = open(filename, "w+")
        # print>>dump, summary
        dump.write(summary)
        dump.close()
        
        # dump all the statistics to the corresponding stats_cls.txt file
        # filename = os.path.join("./results", self.t_sha, "stats_%s.txt" % self.cls)
        filename = os.path.join(self.t_sha, "stats_%s.txt" % self.cls)
        dump = open(filename, "w+")
        # print>>dump, "%.6f " * 21 \
        dump.write("%.6f " * 21 \
                % (self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, \
                   self.recall, self.precision, self.F1, self.FAR, \
                   self.MT, self.PT, self.ML, self.tp, self.fp, self.fn, self.id_switches, self.fragments, \
                   self.n_gt, self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories))
        dump.close()
        
        # write description of statistics to description.txt
        # filename = os.path.join("./results", self.t_sha, "description.txt")
        filename = os.path.join(self.t_sha, "description.txt")
        dump = open(filename, "w+")
        # print>>dump, "MOTA", "MOTP", "MOTAL", "MODA", "MODP", "recall", "precision", "F1", "FAR",
        # print>>dump, "MT", "PT", "ML", "tp", "fp", "fn", "id_switches", "fragments",
        # print>>dump, "n_gt", "n_gt_trajectories", "n_tr", "n_tr_trajectories"
        dump.write("MOTA" + "MOTP" + "MOTAL" + "MODA" + "MODP" + "recall" + "precision" + "F1" + "FAR")
        dump.write("MT" + "PT" + "ML" + "tp" + "fp" + "fn" + "id_switches" + "fragments")
        dump.write("n_gt" + "n_gt_trajectories" + "n_tr" + "n_tr_trajectories")


def evaluate():

    classes = []
    for c in ("car", "people"):
        e = trackingEvaluation(cls=c)
        # load tracker data and check provided classes
        try:
            if not e.loadTracker():
                continue
    
            classes.append(c)
        except:
            break
        # load groundtruth data for this class
        if not e.loadGroundtruth():
            raise ValueError("Ground truth not found.")
        # sanity checks
        if len(e.groundtruth) is not len(e.tracker):

            return False
        # create needed directories, evaluate and save stats
        if e.compute3rdPartyMetrics():
            e.saveToStats()
          

    # finish
    if len(classes)==0:
        return False
    return True

if __name__ == "__main__":
    success = evaluate()

