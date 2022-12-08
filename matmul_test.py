import time
import torch
import torch.profiler as profiler
import argparse
# define tensor_x as a matrix 1664*1664
tensor_x = torch.rand(1664, 1664, device="cuda")
# define tensors_y as 10 128x128 matrices
tensors_y = [torch.rand(128, 128, device="cuda") for _ in range(10)]
activity_groups = [
    profiler.ProfilerActivity.CUDA,
    profiler.ProfilerActivity.CPU,
]
nwarmup = 100
def func_for_profile_mode1():
    # Test A
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    Ar = torch.matmul(tensor_x, tensor_x)
    time.sleep(1e-3)
    # Test B
    Ar = torch.matmul(tensor_x, tensor_x)
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])

def func_for_profile_mode2():
    # Test B
    Ar = torch.matmul(tensor_x, tensor_x)
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    time.sleep(1e-3)
    # Test A
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    Ar = torch.matmul(tensor_x, tensor_x)



def func_for_execution_time_measurement_mode1():
    # warmup
    for i in range(nwarmup):
        for j in range(10):
            torch.matmul(tensors_y[j], tensors_y[j])
        Ar = torch.matmul(tensor_x, tensor_x)

    torch.cuda.synchronize()

    # Test A
    t0 = time.time_ns()
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    Ar = torch.matmul(tensor_x, tensor_x)
    torch.cuda.synchronize()
    t1 = time.time_ns()
    print("exeuction time for small matrix multiplication: %.4fms" % ((t1 - t0) / 1e6))

    torch.cuda.synchronize()

    # Test B
    t2 = time.time_ns()
    Ar = torch.matmul(tensor_x, tensor_x)
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    torch.cuda.synchronize()
    t3 = time.time_ns()
    print("exeuction time for large matrix multiplication: %.4fms" %
          ((t3 - t2) / 1e6))


def func_for_execution_time_measurement_mode2():
    # warmup
    for i in range(nwarmup):
        for j in range(10):
            torch.matmul(tensors_y[j], tensors_y[j])
    torch.cuda.synchronize()

    # Test B
    torch.cuda.synchronize()
    t2 = time.time_ns()
    Ar = torch.matmul(tensor_x, tensor_x)
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    torch.cuda.synchronize()
    t3 = time.time_ns()
    print("exeuction time for big matrix multiplication: ", (t3 - t2) / 1e6)

    torch.cuda.synchronize()
    
    # Test A
    t0 = time.time_ns()
    for j in range(10):
        torch.matmul(tensors_y[j], tensors_y[j])
    Ar = torch.matmul(tensor_x, tensor_x)
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
