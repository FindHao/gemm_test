import time
import torch
import torch.profiler as profiler

# define A as a matrix 1600*1600
A = torch.rand(16, 1600, device="cuda")
# define B as 10 100x100 matrices
B = torch.rand(10, 128, 128, device="cuda")
Br = [None] * 10
activity_groups = [
    profiler.ProfilerActivity.CUDA,
    profiler.ProfilerActivity.CPU,
]
nwarmup = 100


def work():
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])

    time.sleep(1e-3)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)




def work2():
    # warmup
    for i in range(nwarmup):
        for j in range(10):
            # compute the product of A and B[j]
            torch.matmul(B[j], B[j])
    torch.cuda.synchronize()

    time.sleep(1e-3)
    torch.cuda.synchronize()
    t2 = time.time_ns()
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    torch.cuda.synchronize()
    t3 = time.time_ns()
    print("time2: ", (t3 - t2) / 1e6)



    torch.cuda.synchronize()

    t0 = time.time_ns()
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)
    torch.cuda.synchronize()
    t1 = time.time_ns()
    print("time1: ", (t1 - t0) / 1e6)




def profile_step():
    # use torch profiler to profile the code
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1, repeat=1),
        record_shapes=True,
        activities=activity_groups,
        on_trace_ready=profiler.tensorboard_trace_handler("./log")
    ) as prof:

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for i in range(nwarmup + 1):
            work()
            # Need to sync here to match run_one_step()'s timed run.
            torch.cuda.synchronize()
            prof.step()


# work2()
profile_step()
