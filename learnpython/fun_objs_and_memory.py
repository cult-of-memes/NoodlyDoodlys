import os
import multiprocessing as mp
from multiprocessing.connection import Connection as mpConn
from time import perf_counter, sleep
from typing import Optional,Tuple,Callable,Union
from math import atan2
from contextlib import contextmanager

##
# import third-party packages
# get psutil via your preferred python package manager.
# on your command terminal you can use one of the following:
#   * `pip install psutil`
#   * `conda install psutil`
import psutil # psutil is a third party package

# Global constants that may be shared across all class/func invocations
SCREEN_RANGE = 500,800
IMAGE_DOMAIN = (-3.1,1.1),(-1.25,1.25)

def make_scales(
        domain_ij:Tuple[Tuple[Union[int,float]]],
        range_ij:Tuple[Tuple[Union[int,float]]]
)->Tuple[Tuple[float,float],Tuple[int,int]]:
    """A helper function to ensure consistent implementation of simple transform scaler creation."""
    di_span = domain_ij[0][1]-domain_ij[0][0] or 1.
    dj_span = domain_ij[1][1]-domain_ij[1][0] or 1.
    ri_span = range_ij[0][1]-range_ij[0][0] or 1.
    rj_span = range_ij[1][1]-range_ij[1][0] or 1.
    return (di_span/ri_span,dj_span/rj_span),(round(ri_span/di_span),round(rj_span/dj_span))

def simple_coord_xfrm_wrapper(
        d2r_scales:Optional[Tuple[float,float]]=None,
        r2d_scales:Optional[Tuple[float,float]]=None
)->Tuple[Callable[[float,float],Tuple[int,int]],Callable[[int,int],Tuple[float,float]]]:
    """A utility function that configures the mapping between
    screen coordinates and an abstract mathematical plane according
    to the plane's resolution (computed using the values in IMAGE_DOMAIN) relative
    to the resolution of the screen (computed using the values in SCREEN_RANGE).

    "simple_coord_xfrm_wrapper" should be read as "simple coordinate transform wrapper"

    TL;DR:
    We are creating a function to compute a coordinate transformation from some domain to some range,
    and the coresponding inverse function that computes the transformation from range back to the domain.

    The transform of domain -> range is NOT "1-to-1", but it IS "onto".
    The transform of range -> domain IS "1-to-1", and is NOT "onto".

    Note: While normal euclidean coordinates are expressed as the ordered pair `(x,y)`,
    this function uses a more linear algebra approach and maps coordinates as `(i,j)`
        where `i` iterates over rows (the `y` coordinate)
        and j iterates over the columns (the `x` coordinate)
    :param d2r_scales: OPTIONAL;
                       A 2 element tuple of floating point values that represents the scaling factor for
                       transforming points on the abstract plane to the pixels of our image.
                       DEFAULT:
                            (IMAGE_DOMAIN[0][1]-IMAGE_DOMAIN[0][0])/SCREEN_RANGE[0],\
                            (IMAGE_DOMAIN[1][1]-IMAGE_DOMAIN[1][0])/SCREEN_RANGE[1]
    :type d2r_scales: Tuple[int,int]
    :param r2d_scales: OPTIONAL;
                       A 2 element tuple of floating point values that represents the scaling factor for
                       transforming the pixels of our image to the points on the abstract plane.
                       DEFAULT:
                            SCREEN_RANGE[0]/(IMAGE_DOMAIN[0][1]-IMAGE_DOMAIN[0][0]),\
                            SCREEN_RANGE[1]/(IMAGE_DOMAIN[1][1]-IMAGE_DOMAIN[1][0])
    :type r2d_scales: Tuple[int,int]
    :return:
    :rtype:
    """

    def domain2range(i:float,j:float,**kwargs):
        return int(i*d2r_i),int(j*d2r_j)

    def range2domain(i:int,j:int,**kwargs)->Tuple[float,float]:
        return float(i*r2d_i),float(j*r2d_j)

    # configure local scope variables according to function args and
    if d2r_scales is None:
        d2r_scales = make_scales(IMAGE_DOMAIN,((0,SCREEN_RANGE[0]),(0,SCREEN_RANGE[1])))
    if r2d_scales is None:
        r2d_scales = make_scales(((0,SCREEN_RANGE[0]),(0,SCREEN_RANGE[1])),IMAGE_DOMAIN)
    # precompute constant multipliers from our scaling factors
    # these constants will be available to the inner functions
    # that share the scope created by `coordinate_transform_wrapper`
    # but are unvailable to external entities so we don't have to
    # worry about namespace collisions.
    d2r_i= d2r_scales[0]
    d2r_j= d2r_scales[1]
    r2d_i = r2d_scales[0]
    r2d_j = r2d_scales[1]
    del d2r_scales,r2d_scales
    return domain2range,range2domain

# Meow, imagine several other wrapper functions with actually interesting or
# scientifically usefull transformation functions that share a common interface
# to the wrapper function above.
# ...

class SimpleCoordXfrmClass:
    """A class implementation of the "simple coordinate transform wrapper" function above.
    """
    def __init__(self,
                 d2r_scales:Optional[Tuple[float,float]]=None,
                 r2d_scales:Optional[Tuple[float,float]]=None) -> None:
        super().__init__()
        if d2r_scales is None:
            d2r_scales = make_scales(IMAGE_DOMAIN,((0,SCREEN_RANGE[0]),(0,SCREEN_RANGE[1])))
        if r2d_scales is None:
            r2d_scales = make_scales(((0,SCREEN_RANGE[0]),(0,SCREEN_RANGE[1])),IMAGE_DOMAIN)
        # precompute constant multipliers from our scaling factors
        # these constants will be available to the inner functions
        # that share the scope created by `coordinate_transform_wrapper`
        # but are unvailable to external entities so we don't have to
        # worry about namespace collisions.
        self.d2r_i = d2r_scales[0]
        self.d2r_j = d2r_scales[1]
        self.r2d_i = r2d_scales[0]
        self.r2d_j = r2d_scales[1]

    def domain2range(self, i: float, j: float, **kwargs):
        return int(i * self.d2r_i), int(j * self.d2r_j)

    def range2domain(self, i: int, j: int, **kwargs) -> Tuple[float, float]:
        return float(i * self.r2d_i), float(j * self.r2d_j)

# Meow, like we did with the functions, image several different kinds of classes
# dedicated to different kinds of transformations but with a common interface
# to the class definition above.
# ...

def child_proc_measure_class_mem(pip:mpConn,shared_rand_coords:Tuple[float,...]):
    with pip:
        transforms = [
            SimpleCoordXfrmClass(
                *make_scales(((-y,y),(-x,x)),((0,SCREEN_RANGE[0]*angle),(0,SCREEN_RANGE[1]*angle)))
            ) for y,x,angle in shared_rand_coords
        ]
        proc = psutil.Process(os.getpid())
        mem = proc.memory_full_info()
        pip.send(("class",mem))

def child_proc_measure_func_mem(pip:mpConn,shared_rand_coords:Tuple[float,...]):
    with pip:
        start = perf_counter()
        transforms = [
            simple_coord_xfrm_wrapper(
                *make_scales(((-y,y),(-x,x)),((0,SCREEN_RANGE[0]*angle),(0,SCREEN_RANGE[1]*angle)))
            ) for y,x,angle in shared_rand_coords
        ]
        proc = psutil.Process(os.getpid())
        mem = proc.memory_full_info()
        sleep(9-(perf_counter()-start)) # give time to watch the "my_beach_ball"... :{P
        pip.send(("func",mem))

@contextmanager
def multi_proc_manager(num_procs:int,targets:tuple,is_duplex:bool,is_daemon,inpt_arr:tuple,in_pipes:list,procs:list):
    num_procs -= 1
    in_pip,out_pip = mp.Pipe(duplex=is_duplex)
    with in_pip:
        with out_pip:
            proc = mp.Process(target=targets[num_procs],args=(out_pip,inpt_arr),daemon=is_daemon)
            procs.append(proc)
            in_pipes.append(in_pip)
            if num_procs:
                with multi_proc_manager(num_procs,targets,is_duplex,is_daemon,inpt_arr,in_pipes,procs) as pipes:
                    yield pipes
            else:
                try:
                    for proc in procs:
                        proc.start()
                    yield in_pipes
                finally:
                    proc.join()
                    proc.close()

def make_activity_indicator():
    from itertools import permutations
    ret = [
        "v( | )v", "v( | )v", "v( | )v", "v( | )v", "v( | )v", "v( | )v",
        "<( | )>", "<( | )>", "<( | )>", "<( | )>", "<( | )>", "<( | )>",
        "^( | )^", "^( | )^", "^( | )^", "^( | )^", "^( | )^", "^( | )^",
    ]
    shake = [
        "( | )", "( | )", "(  ))", "(  ))", "( | )", "( | )",
        "((  )", "((  )", "( | )", "( | )", "(  ))", "(  ))",
        "( | )", "( | )", "((  )", "((  )",
    ]
    flail = [("^","^")]+list(tpl for tpl in permutations("^v<>",2) if tpl[0]!=">" and tpl[1]!="<")
    combo_template = "{l}{b}{r}"
    for l,r in flail:
        for b in shake:
            elem = combo_template.format(l=l,b=b,r=r)
            ret.append(elem)
    return ret

def main():
    my_beach_ball = make_activity_indicator()
    prog_pos = 0
    start = perf_counter()
    start_len = 10
    system_mem_checks = []
    sys_mem_msgs = [
        "memory at init",
        "memory after first append",
        "memory after creating shared_inpt_arr",
        "memory after creating process context",
    ]
    util_proc = psutil.Process(os.getpid())
    system_mem_checks.append(util_proc.memory_full_info())
    system_mem_checks.append(util_proc.memory_full_info())
    inpt_arr = [(i/1000,j/1000) for i in range(-100,100+1) for j in range(-100,101)]
    inpt_arr = [(y,x,atan2(y,x)) for y,x in inpt_arr]
    system_mem_checks.append(util_proc.memory_full_info())
    with multi_proc_manager(2,(child_proc_measure_class_mem,child_proc_measure_func_mem),False,True,inpt_arr,[],[]) as pipes:
        system_mem_checks.append(util_proc.memory_full_info())
        start_len = len(system_mem_checks)
        start = perf_counter()
        while (len(system_mem_checks)-start_len)<len(pipes) and perf_counter()-start<10:
            for idx,p in enumerate(pipes):
                print("\r"+my_beach_ball[prog_pos],end="")
                prog_pos = (prog_pos+1)%len(my_beach_ball)
                p: mpConn
                if p.poll(.15):
                    check_type,mem_use = p.recv()
                    sys_mem_msgs.append(check_type)
                    system_mem_checks.append(mem_use)
                    start = perf_counter()
    print()
    sys_mem_msgs.append("memory after exiting process context")
    system_mem_checks.append(util_proc.memory_full_info())
    KiB,MiB,GiB = 2**10,2**20,2**30
    longest_filed = len("peak_nonpaged_pool")
    for msg,use in zip(sys_mem_msgs,system_mem_checks):
        use_str = []
        if hasattr(use,"_fields"):
            for k in ("uss","rss"):
                v = getattr(use,k)
                if v>=GiB:
                    v /= GiB
                    unit = "KiB"
                elif v>= MiB:
                    v /= MiB
                    unit = "MiB"
                elif v>= KiB:
                    v /= KiB
                    unit = "GiB"
                # these next few lines are because string formatting won't align on decimal
                # the way I want.
                val = str(round(v,3))
                whole,frac = val.split(".")
                whole = " "*(3-len(whole))+whole
                frac += " "*(3-len(frac))
                use_str.append(f"\n\t{k:>{longest_filed}}: {whole+'.'+frac} {unit}")
        else:
            print(f"{msg}: {type(use)=}")
        print(f"{msg}:{''.join(use_str)}")

if __name__ == '__main__':
    main()

example_output = """\
^( | )>
memory at init:
	               uss:   8.297 MiB
	               rss:  15.836 MiB
memory after first append:
	               uss:   8.297 MiB
	               rss:  15.871 MiB
memory after creating shared_inpt_arr:
	               uss:  17.621 MiB
	               rss:  25.219 MiB
memory after creating process context:
	               uss:  17.367 MiB
	               rss:  25.113 MiB
class:
	               uss:  26.543 MiB
	               rss:  34.195 MiB
func:
	               uss:  65.406 MiB
	               rss:  72.812 MiB
memory after exiting process context:
	               uss:  17.367 MiB
	               rss:  25.152 MiB

Process finished with exit code 0
"""
