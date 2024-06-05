# no parser
from multiprocessing import Process,Queue,Lock
import pygraphviz as pgv
import struct
import mmap
import os
from time import sleep
from energy_count.ops_count import *
import networkx as nx
# temporally
countable_ops = {
                 "MmBackward": matrix_mul,
                 "BmmBackward0": batch_matrix_mul,
                 "ThnnConv2DBackward": conv2d,
                 "CudnnConvolutionBackward": conv2d,
                 "MkldnnConvolutionBackward": conv2d,
                 "AddBackward0": tensor_add,
                 "MulBackward0": matrix_dot,
                 "SqrtBackward": tensor_pow,
                 "PowBackward0": tensor_pow,
                 "DivBackward0": matrix_dot,
                 "SubBackward0": tensor_add,
                 "AddmmBackward": matrix_mul,
                 "FftR2CBackward": fft_r2c,
                 "FftC2RBackward": fft_c2r,
                 "EmbeddingBackward": embedding,
                 }

shape_change_ops = {
    "CatBackward": cat,
    "SqueezeBackward1": squeeze,
    "UnsqueezeBackward0": unsqueeze,
    "SumBackward1": mean_sum,
    "MeanBackward1": mean_sum,
    "MaxPool2DWithIndicesBackward": maxpool_2d,
    "AvgPool2DBackward": maxpool_2d,
    "SelectBackward": squeeze,
    "StackBackward": stack,
}

status_change_ops = {
    "CudnnBatchNormBackward": norm,
    "atanBackward": atan,
}

logger_file = "mac_ac.dat"
shared_file = "route.dat"
len_nid = 13
module_lsar_dict = {}


def lsar_register(net):
    m_dict = dict(net.named_modules())
    for key in m_dict.keys():
        if key == "":
            continue
        m = m_dict[key]
        m.register_forward_hook(ops_hook_fn(key+".weight"))


# this function is especially prepared for lasr and ac attr
def ops_hook_fn(module_name):
    def hook(m,inputs,outputs):
        import torch
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_mac = max_v.dtype == torch.int64 or not torch.floor(max_v) == max_v
        ran = max_v.detach().cpu().numpy().astype(int)
        lsar = 0
        for i in range(1, ran + 1):
            lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        if lsar == 0:
            lsar = inputs.count_nonzero() / inputs.numel()
        module_lsar_dict[module_name] = [float(lsar),is_mac]
    return hook


def parse_label(lbl_info):
    # need to be reconstructed using re
    lbl_info = lbl_info.replace(" ","")
    fullname = lbl_info.split("-")[0]
    lbl_info = lbl_info.replace(fullname,"")
    lbl_info = lbl_info.replace("-","")
    attrs = {}
    l_attrs = lbl_info.split(":")
    attrs[l_attrs[0]] = None
    last_key = l_attrs[0]
    l_attrs = l_attrs[1:]
    try:
        for attr in l_attrs:
            last_key.replace(" ","")
            # special key dealt
            if last_key == "dim":
                lf = attr.find("(")
                rf = attr.find(")")
                if lf == -1 or rf == -1:
                    dim = attr[0]
                    attrs[last_key] = int(dim)
                    last_key = attr[1:]
                else:
                    dim = attr[lf+1:rf].replace(",","")
                    attrs[last_key] = int(dim)
                    last_key = attr[rf+1:]
                continue
            elif last_key == "keepdim":
                v = attr.find("True")
                attrs[last_key] = attr[v:v+4]
                last_key = attr[v+4:]
                if v == -1:
                    v = attr.find("False")
                    attrs[last_key] = attr[v:v + 5]
                    last_key = attr[v + 5:]
                continue
            elif last_key == "groups":
                v = int(attr[0])
                attrs[last_key] = v
                last_key = attr[1:]
                continue
            if "None" in attr:
                attrs[last_key] = None
                last_key = attr.split("None")[-1]
                attrs[last_key] = None
            elif "False" in attr:
                attrs[last_key] = False
                v = attr.find("False")
                last_key = attr[v+5:]
            elif "True" in attr:
                attrs[last_key] = False
                v = attr.find("True")
                last_key = attr[v+4:]
            else:
                lf = attr.find("(")
                rf = attr.find(")")
                if lf == -1 or rf == -1:
                    lf = attr.find("[")
                    rf = attr.find("]")
                    info = attr[lf:rf+1]
                elif lf == rf - 1:
                    info = None
                else:
                    info = [int(i) for i in attr[lf+1:rf].split(",")]
                attrs[last_key] = info
                last_key = attr[rf+1:]
                if last_key != "":
                    attrs[last_key] = None
    except:
        print(f"parse exception at {lbl_info}")
        pass
    return fullname,attrs
    pass


def create_global_graph(model,inputs: tuple):
    from torchviz import make_dot
    outputs = model(*inputs)
    dot = make_dot(outputs,params=dict(model.named_parameters()),show_attrs=True)
    # dot.view()
    # exit(0)
    return dot.source,outputs


def create_forward_mission(q,graph_str,*args):
    p = Process(target=single_forward,args=(q,graph_str,*args,))
    p.start()
    # single_forward(graph_str,*args)
    pass


def create_log_mission(q):
    p = Process(target=asynchronous_logger,args=(q,))
    p.start()


def single_forward(logger_q,graph_str,st,ops1,ops2,num,par_num,flock,lsar=1.0,is_mac=False,splitted=False):
    graph_raw = pgv.AGraph(graph_str)
    chl_edges = graph_raw.out_edges(st)
    num_par_edges = graph_raw.in_degree(st)
    if len(chl_edges) > 1 and splitted:
        chl_edges = [chl_edges[num - par_num - 1]]
    if num_par_edges == 0 or num_par_edges > 1:  # start node
        num_par_edges = 1
    route_st = st
    mm = p_mmap(shared_file)
    while len(chl_edges) == 1 and num_par_edges == 1:
        lbl_info = graph_raw.get_node(st).attr['label']
        fullname,attrs = parse_label(lbl_info)
        # if "self_sizes" in attrs.keys() and fullname == "ViewBackward":
        #     ops2 = attrs['self_sizes']
        if "self_sizes" in attrs.keys() and ops2 is None:
            ops2 = attrs['self_sizes']
        if fullname in shape_change_ops.keys():
            ops1 = None
            sc_fn = shape_change_ops[fullname]
            attrs['num_par_edges'] = graph_raw.in_degree(st)
            try:
                ops2 = sc_fn(ops2,attrs)
            except:
                pass
                # ViewBackward would give it
        elif fullname in status_change_ops.keys():
            sc_fn = status_change_ops[fullname]
            lsar,is_mac = sc_fn()
        elif fullname in countable_ops.keys() and not splitted:
            count_fn = countable_ops[fullname]
            if "Conv" in fullname:
                ops1.append(attrs['stride'])
                ops1.append(attrs['padding'])
            elif ops2 is None:
                ops2 = ops1
            attrs['mac'] = is_mac
            logger_q.put([fullname,ops1,ops2])
            mac,ac,ops1,ops2 = count_fn(ops1,ops2,attrs)
            mac *= lsar
            ac *= lsar
            logger_q.put([0, mac, ac])
            splitted = False
        elif "NativeLayerNormBackward" in fullname and not splitted:
            mac,ac = lnorm(ops2)
            lsar = 1.0
            is_mac = True
            logger_q.put([0,mac,ac])
            splitted = False
        st = chl_edges[0][1]
        chl_edges = graph_raw.out_edges(st)
        num_par_edges = graph_raw.in_degree(st)
        ops = ops2 if ops2 is not None else ops1
        route_info = (route_st,st,"0",ops,lsar,is_mac)  # reset status, status here can be mac or ac
        write_route_to_file(mm,num,*route_info)
    if num_par_edges > 1:
        # when you find this we are in one of the parent of this node
        # read route files,search all,find one route that meet here
        # need a synchonize here
        p_munmap(mm)
        ops = ops2 if ops2 is not None else ops1
        loop_count = 0
        s_time = 0.1
        fullname, _ = parse_label(graph_raw.get_node(st).attr['label'])
        while True:
            status,new_ops,new_lsar,new_is_mac,flag = multi_in_degree_node_synchronous(num,st,ops,flock)
            loop_count += 1
            s_time += 0.05
            if loop_count > 15:
                break
            if not flag:
                # when the network is big, sleep time needs to be long, otherwise cpu burden is high
                sleep(s_time)
            else:
                break
        # synchronous successfully
        # which one is first to reach that, create a new process, else exit directly
        pass
        # after find, create a new process, then both parent process exit
        if status == "0":  # first one to read that, get the permission to create new process
            new_ops = [*new_ops]
            lsar = min(lsar,new_lsar)
            is_mac = new_is_mac and is_mac
            new_args = (st.__str__(),new_ops,ops,num,num,flock,lsar,is_mac,False)
            create_forward_mission(logger_q,graph_str,*new_args)
        else:
            mm = p_mmap(shared_file)
            num_pc = read_num_routes_from_file(mm)
            logger_q.put([1, num_pc, 0])
    elif len(chl_edges) > 1:
        fullname,attrs = parse_label(graph_raw.get_node(st).attr['label'])
        if "SplitBackward" in fullname:
            dim = attrs['dim']
            # ops2 = attrs['self_sizes']
            ops2[dim] //= len(chl_edges)
        elif fullname in countable_ops.keys():
            count_fn = countable_ops[fullname]
            if "Conv" in fullname:
                ops1.append(attrs['stride'])
                ops1.append(attrs['padding'])
            elif ops2 is None:
                ops2 = ops1
            attrs['mac'] = is_mac
            mac, ac, ops1, ops2 = count_fn(ops1, ops2,attrs)
            mac *= lsar
            ac *= lsar
            logger_q.put([0, mac, ac])
        elif "NativeLayerNormBackward" in fullname:
            mac, ac = lnorm(ops2)
            lsar = 1.0
            is_mac = True
            logger_q.put([0, mac, ac])
        # create all chl
        flock.acquire()
        total = read_num_routes_from_file(mm)
        write_info_into_file(mm,struct.pack("I",(len(chl_edges)+total)),0)
        st_str = st.__str__()
        flock.release()
        for sub_id,chl_edge in enumerate(chl_edges):
            ops = ops2 if ops2 is not None else ops1
            route_infos = (st_str,st_str,"0",ops,lsar,is_mac)
            write_route_to_file(mm,sub_id + total + 1,*route_infos)
            args = (st_str,ops1,ops2,sub_id + total + 1,total,flock,lsar,is_mac,True)
            create_forward_mission(logger_q,graph_str,*args)
        num_pc = read_num_routes_from_file(mm)
        logger_q.put([1, num_pc, 0])
        p_munmap(mm)
    # normal exit
    if len(chl_edges) == 0:
        mm = p_mmap(shared_file)
        num_pc = read_num_routes_from_file(mm)
        logger_q.put([2,num_pc,0]) # finish
    exit(0)


def p_munmap(mm):
    mm.flush()
    mm.close()


def p_mmap(file_name):
    with open(file_name,"r+b") as f:
        st = os.fstat(f.fileno())
        length = st.st_size
        mm = mmap.mmap(f.fileno(),length,access=mmap.ACCESS_WRITE)
    return mm


def multi_in_degree_node_synchronous(id,nd_name,ops,flock):
    # spinlock implementation
    mm = p_mmap(shared_file)
    flock.acquire()
    num_routes = read_num_routes_from_file(mm)
    flock.release()
    for i in range(1,num_routes):
        flock.acquire()
        start,cur,status,shape,lsar,is_mac = read_route_from_file(mm,i)
        if start is None:
            flock.release()
            p_munmap(mm)
            return "1", None,lsar,is_mac, False
        if cur == nd_name and status == "0":  # 顺序全局固定!
            route_info = (start,cur,"1",ops,lsar,is_mac)
            if id != i:
                # take ops, and change self status into "1" as well!
                s_st,s_cur,s_sta,s_shape,s_lsar,s_is_mac = read_route_from_file(mm,id)
                s_sta = "1"
                s_route_info = (s_st,s_cur,s_sta,s_shape,s_lsar,s_is_mac)
                write_route_to_file(mm,id,*s_route_info)
                write_route_to_file(mm, i, *route_info)
                p_munmap(mm)
                flock.release()
                return status,shape,lsar,is_mac,True
        elif cur == nd_name and status == "1":
            # someone has acquire my ops
            p_munmap(mm)
            flock.release()
            return status,None,lsar,is_mac,True
        flock.release()
    p_munmap(mm)
    return "1", None,1,True,False


def write_route_to_file(mm,id,*infos):
    start,cur,status,shape,lsar,is_mac = infos
    item_size = len_nid * 2 + 1 + 8 + 6 * 4
    dim = len(shape)
    fmt = "f" + "I" * (dim + 2)
    if is_mac:
        is_mac = 1
    else:
        is_mac = 0
    pck_shape = struct.pack(fmt,lsar,is_mac,dim,*shape)
    brt_info = (start + cur + status).encode()
    com_info = brt_info + pck_shape
    offset = (id - 1) * item_size + 4
    write_info_into_file(mm,com_info,offset)
    pass


def write_info_into_file(mm,binfo,offset):
    mm[offset:offset + len(binfo)] = binfo
    mm.flush()


def read_route_from_file(mm,route_num):
    item_sz = len_nid * 2 + 1 + 8 + 6 * 4
    offset = item_sz * (route_num - 1) + 4
    route = mm[offset:offset+item_sz]
    start = route[:len_nid].decode()
    cur = route[len_nid:len_nid*2].decode()
    status = route[len_nid*2:len_nid*2+1].decode()
    lsar, is_mac = struct.unpack("fI", route[len_nid*2+1:len_nid*2+9])
    bdim = route[len_nid*2+9:len_nid*2+13]
    fmt = "I"
    dim = struct.unpack(fmt,bdim)[0]
    if dim > 5:
        return None,None,None,None,None,None
    bshape = route[len_nid*2+13:len_nid*2+13+dim*4]
    fmt = "I" * dim
    shape = struct.unpack(fmt,bshape)
    is_mac = is_mac == 1
    return start,cur,status,shape,lsar,is_mac


def read_num_routes_from_file(mm):
    num_routes = struct.unpack("I",mm[:4])[0]
    return num_routes


def init_synchronous_file(fpath,num_routes):
    with open(fpath,"w+b") as f:
        empty_content = "1" * 100000
        f.write(empty_content.encode())
    f.close()
    fmt = "I"
    pck_data = struct.pack(fmt,num_routes)
    mm = p_mmap(fpath)
    write_info_into_file(mm,pck_data,0)
    p_munmap(mm)


def energy_count(model,model_inputs:tuple):
    lsar_register(model)
    graph_str,outputs = create_global_graph(model, model_inputs)
    graph_raw = pgv.AGraph(graph_str)
    routes = []
    file_lock = Lock()
    count = 0
    for node in graph_raw.nodes():
        if graph_raw.in_degree(node) == 0:
            node_str = node.__str__()
            label = node.attr['label']
            full_name,_ = parse_label(label)
            lf = label.find("(")
            rf = label.find(")")
            ops = label[lf + 1:rf]
            ops1 = [int(ops_s) for ops_s in ops.split(",")]
            # cut all bias branch, remains some doubts here
            if len(ops1) == 1 and "layernorm.weight" not in full_name:
                graph_raw = clear_bias_route(graph_raw, node)
                continue
            elif full_name[0] == '(' and full_name[-1] == ')':
                if not node.attr['fillcolor'] == "lightblue":
                    graph_raw = clear_bias_route(graph_raw,node)
                    continue
            ops2 = None
            k_full_name = full_name.split("(")[0]
            lsar_info = [1.0,True]
            if "embedding" in k_full_name:
                ops2 = list(model_inputs[0].shape)
            elif "weight" in k_full_name:
                if k_full_name in module_lsar_dict.keys():
                    lsar_info = module_lsar_dict[k_full_name]
                else:
                    lsar_info = [1.0,True]
            elif node.attr['fillcolor'] == "lightblue":
                ops2 = ops1
            count += 1
            routes.append((node_str, ops1, ops2, count, count, file_lock, *lsar_info,False))
    # cycle removal preprocess: especially for snn
    # graph_raw.draw("file.pdf",prog="dot")
    try:
        NXG = nx.nx_agraph.from_agraph(graph_raw)
        cycle = nx.algorithms.cycles.find_cycle(NXG)
    except:
        pass
    else:
        for edge in cycle:
            end_node_lbl = graph_raw.get_node(edge[1]).attr['label']
            end_node_lbl = end_node_lbl.strip(" ")
            if end_node_lbl[0] == "(" and end_node_lbl[-1] == ")":
                # special cycle case for snn
                graph_raw.remove_node(edge[1])
                print(f"remove {end_node_lbl}")
                break
    try:
        NXG = nx.nx_agraph.from_agraph(graph_raw)
        nx.algorithms.cycles.find_cycle(NXG)
    except:
        graph_str = graph_raw.to_string()
        graph_raw.draw("file.pdf",prog="dot")
        init_synchronous_file(shared_file, len(routes))
        logger_q = Queue()
        create_log_mission(logger_q)
        for no, route in enumerate(routes):
            create_forward_mission(logger_q, graph_str, *route)
        pass
    else:
        print("Graph have a cycle we can not handle! Resist to process")
    open(logger_file, "w").close()
    while os.path.exists(logger_file):
        continue
    return outputs
    # print("Graph has cycle:", cycle)
    # graph_str = graph_raw.to_string()
    # init_synchronous_file(shared_file, len(routes))
    # logger_q = Queue()
    # create_log_mission(logger_q)
    # for no, route in enumerate(routes):
    #     create_forward_mission(logger_q, graph_str, *route)


def clear_bias_route(graph_raw,node):
    rm_list = [node]
    while len(rm_list) > 0:
        new_node = rm_list.pop()
        chl_edges = graph_raw.out_edges(new_node)
        for chl_edge in chl_edges:
            chl_node = chl_edge[1]
            if graph_raw.in_degree(chl_node) <= 1:
                rm_list.append(chl_node)
        graph_raw.remove_node(new_node)
    return graph_raw
    pass


if __name__ == "__main__":
    pass
'''
TODO LIST: 
1. file sychro done
2. process synchro done(sleep time needs to be a little bit longer), there is still problems here
3. count fn done
4. logger (sum energy count, display processing duration) done

5. support conv ops Done
6. multi in-degree(more than 2, this normally happened at cat ops) node synchro support, Done

7. support lasr,ac parameter (critical important)
8. support more ops (fft,rfft,div,sub)
'''

