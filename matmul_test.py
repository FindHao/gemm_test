import time
import torch
import torch.profiler as profiler
import argparse
# define A as a matrix 1664*1664
A = torch.rand(16, 1664, device="cuda")
# define B as 10 128x128 matrices
B = torch.rand(10, 128, 128, device="cuda")
Br = [None] * 10
activity_groups = [
    profiler.ProfilerActivity.CUDA,
    profiler.ProfilerActivity.CPU,
]
nwarmup = 100
def func_for_profile_mode1():
    # Test A
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)
    time.sleep(1e-3)
    # Test B
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])

def func_for_profile_mode2():
    # Test B
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    time.sleep(1e-3)
    # Test A
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)



def func_for_execution_time_measurement_mode1():
    # warmup
    for i in range(nwarmup):
        for j in range(10):
            # compute the product of A and B[j]
            torch.matmul(B[j], B[j])
    torch.cuda.synchronize()

    # Test A
    t0 = time.time_ns()
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)
    torch.cuda.synchronize()
    t1 = time.time_ns()
    print("exeuction time for small matrix multiplication: ", (t1 - t0) / 1e6)

    torch.cuda.synchronize()

    # Test B
    torch.cuda.synchronize()
    t2 = time.time_ns()
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    torch.cuda.synchronize()
    t3 = time.time_ns()
    print("exeuction time for big matrix multiplication: ", (t3 - t2) / 1e6)
    

def func_for_execution_time_measurement_mode2():
    # warmup
    for i in range(nwarmup):
        for j in range(10):
            # compute the product of A and B[j]
            torch.matmul(B[j], B[j])
    torch.cuda.synchronize()

    # Test B
    torch.cuda.synchronize()
    t2 = time.time_ns()
    Ar = torch.matmul(A, A)
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    torch.cuda.synchronize()
    t3 = time.time_ns()
    print("exeuction time for big matrix multiplication: ", (t3 - t2) / 1e6)

    torch.cuda.synchronize()
    
    # Test A
    t0 = time.time_ns()
    for j in range(10):
        # compute the product of A and B[j]
        torch.matmul(B[j], B[j])
    Ar = torch.matmul(A, A)
    torch.cuda.synchronize()
    t1 = time.time_ns()
    print("exeuction time for small matrix multiplication: ", (t1 - t0) / 1e6)


def profile_step(mode=1):
    if mode == 1:
        func_for_profile = func_for_profile_mode1
    else:
        func_for_profile = func_for_profile_mode2
    # use torch profiler to profile the code
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1, repeat=1),
        record_shapes=True,
        activities=activity_groups,
        on_trace_ready=profiler.tensorboard_trace_handler("./log")
    ) as prof:
        for i in range(nwarmup + 1):
            func_for_profile()
            # Need to sync here to match run_one_step()'s timed run.
            torch.cuda.synchronize()
            prof.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", type=int, default=1)
    args = parser.parse_args()
    if args.profile:
        profile_step(mode=args.mode)
    else:
        if args.mode == 1:
            func_for_execution_time_measurement_mode1()
        else:
            func_for_execution_time_measurement_mode2()
